import os
import io
import wandb
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from statistics import mean

# Abstract base class for context-dependent language approaches (DuelingDDQN and TD3)
class CDL:
    def __init__(self):
        self.min_lines = 1
        self.max_lines = 8
        self.configs_to_consider = 250
        self.rng = np.random.default_rng(seed=42)
        
        self.buffer = None
        self.state_dim = 128
        self.valid_lines = set()
        self.name = self.__class__.__name__
            
    def _save(self, approach, problem_instance, language):
        directory = f'ma-cdl/languages/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(language, file)
    
    def _load(self, approach):
        directory = f'ma-cdl/languages/history/{approach.lower()}'
        file_path = os.path.join(directory)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        return language
    
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'{self.__class__.__name__}/{problem_instance.capitalize()}')
        config = wandb.config
        return config
    
    # Log regions to WandB
    def _log_regions(self, problem_instance, title_name, title_data, regions, reward):
        _, ax = plt.subplots()
        problem_instance = problem_instance.capitalize()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Problem Instance: {problem_instance}   {title_name.capitalize()}: {title_data}   Reward: {reward:.2f}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        plt.close()
        wandb.log({"image": wandb.Image(pil_image)})
    
    # Visualize regions that define the language
    def _visualize(self, approach, problem_instance, language):
        plt.clf()
        plt.cla()
        
        for idx, region in enumerate(language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')

        directory = f'ma-cdl/languages/history/{approach.lower()}'
        filename = f'{problem_instance}.png'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close('all')
                
    # Generate configuration under specified constraint
    def _generate_configuration(self, problem_instance):
        self.scenario.reset_world(self.world, self.rng, problem_instance)
        
        rand_idx = self.rng.choice(len(self.world.agents))
        start = self.world.agents[rand_idx].state.p_pos
        goal = self.world.agents[rand_idx].goal.state.p_pos
        obstacles = [obs.state.p_pos for obs in self.world.large_obstacles]

        return start, goal, obstacles
    
    # Create graph from language excluding regions with obstacles
    @staticmethod
    def get_safe_graph(regions, obstacles):
        graph = nx.Graph()

        obstacle_regions = [idx for idx, region in enumerate(regions) if any(region.intersects(obstacle) for obstacle in obstacles)]
        
        # Add nodes to graph
        for idx, region in enumerate(regions):
            if idx in obstacle_regions:
                continue
            centroid = region.centroid
            graph.add_node(idx, position=(centroid.x, centroid.y))

        for idx, region in enumerate(regions):
            if idx in obstacle_regions:
                continue

            for neighbor_idx, neighbor in enumerate(regions):
                if idx == neighbor_idx or neighbor_idx in obstacle_regions:
                    continue
                
                if region.dwithin(neighbor, 4.0000001e-4):
                    graph.add_edge(idx, neighbor_idx)

        return graph
    
    # Populate buffer with dummy transitions
    def _populate_buffer(self):        
        for _ in range(self.dummy_episodes):
            done = False
            num_lines = 0
            while not done:
                num_lines += 1
                action = action_selection()
                reward, next_state, done, _ = self._step(problem_instance, action, num_lines)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
    
    # Overlay new line in the environment
    def _step(self, problem_instance, line, num_lines):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
        
        linestring = CDL.get_shapely_linestring(line)
        valid_lines = CDL.get_valid_lines(linestring)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_lines == self.max_lines:
            done = True
            reward = self.optimizer(regions, problem_instance)
            self.valid_lines.clear()
        
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
        
    """
    Calculate cost of a configuration (i.e. start position, goal position, and obstacle positions)
    with respect to the regions based on the amount of unsafe area (flexibility).
    """
    def _config_cost(self, start, goal, obstacles, regions): 
        def euclidean_distance(a, b):
            return regions[a].centroid.distance(regions[b].centroid)
        
        obstacles_with_size = [Point(obs_pos).buffer(self.obstacle_radius) for obs_pos in obstacles]
    
        graph = CDL.get_safe_graph(regions, obstacles_with_size)
        start_region = CDL.localize(start, regions)
        goal_region = CDL.localize(goal, regions)
        path = []
        try:
            path = nx.astar_path(graph, start_region, goal_region, heuristic=euclidean_distance)
            safe_area = [regions[idx].area for idx in path]
            avg_safe_area = mean(safe_area)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            avg_safe_area = -6
            
        return avg_safe_area
    
    """ 
    Calculate cost of a given problem (i.e. all configurations) 
    with respect to the regions and the given constraints: 
        1. Mean of unsafe area
        2. Variance of unsafe_area
    """
    def optimizer(self, regions, problem_instance):  
        instance_cost = -8
        
        if len(regions) > 1:
            safe_area = []
            efficiency = len(regions)
            for _ in range(self.configs_to_consider):
                start, goal, obstacles = self._generate_configuration(problem_instance)
                config_cost = self._config_cost(start, goal, obstacles, regions)
                safe_area.append(config_cost)
        
            safe_area_mu = mean(safe_area)
            instance_cost = safe_area_mu - 0.2 * efficiency
        
        return instance_cost

    # Train the child model on a given problem instance
    def _train(self, problem_instance, start_state):
        policy_losses = []
        value_losses = []
        returns = []
        best_lines = None
        best_regions = None
        best_reward = -np.inf

        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            action_lst = []
            state = start_state
            while not done:
                num_lines += 1
                action = self._select_action(state)
                reward, next_state, done, regions = self._step(problem_instance, action, num_lines)
                self.buffer.add((state, action, reward, next_state, done))
                policy_loss, value_loss, td_error, tree_idxs = self._learn()
                state = next_state
                action_lst.append(action)

            self.buffer.update_priorities(tree_idxs, td_error)
            self._decrement_exploration()

            value_losses.append(value_loss)
            returns.append(reward)
            avg_value_losses = np.mean(value_losses[-100:])
            avg_returns = np.mean(returns[-100:])
            wandb.log({"Average Value Loss": avg_value_losses})
            wandb.log({"Average Returns": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, 'Episode', episode, regions, reward)
                            
            # Log policy loss if applicable
            if policy_loss is not None:
                policy_losses.append(policy_loss)
                avg_policy_losses = np.mean(policy_losses[-100:])
                wandb.log({"Average Policy Loss": avg_policy_losses})

            if reward > best_reward:
                best_lines = action_lst
                best_regions = regions
                best_reward = reward

        return best_lines, best_regions, best_reward
    
    # Retrieve optimal set lines for a given problem instance from the training phase
    def _generate_optimal_lines(self):        
        self._populate_buffer()
        best_lines, best_regions, best_reward = self._train()
        
        self._log_regions(problem_instance, 'Episode', 'Final', best_regions, best_reward)
        wandb.log({"Final Reward": best_reward})
        wandb.log({"Final Lines": best_lines})
        wandb.finish()  
        
        optim_lines = np.array(best_lines).reshape(-1, 3)   
        return optim_lines  
                
    # Returns regions that defines the language
    def get_language(self):
        print(f'No stored language.')
        print('Generating new language...\n')
        language = self._generate_optimal_lines()
        self._save(language)
        
        return language