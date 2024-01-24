import copy
import Signal8
import numpy as np

from plotter import plot_metrics
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath
from languages.discrete_rl import BasicDQN, CommutativeDQN
from languages.continuous_rl import BasicTD3, CommutativeTD3

class MA_CDL():
    def __init__(self, reward_prediction_type, num_agents, num_large_obstacles, num_small_obstacles, random_state, render_mode, max_cycles=50):
        
        self.env = Signal8.env(
            num_agents=num_agents, 
            num_large_obstacles=num_large_obstacles, 
            num_small_obstacles=num_small_obstacles, 
            render_mode=render_mode,
            max_cycles=max_cycles
            )
        
        scenario = self.env.unwrapped.scenario
        world = self.env.unwrapped.world
        agent_radius = world.agents[0].radius
        obstacle_radius = world.small_obstacles[0].radius 
        
        # Discrete RL
        self.basic_dqn = BasicDQN(scenario, world, random_state)
        self.commutative_dqn = CommutativeDQN(scenario, world, random_state, reward_prediction_type)
        
        # Continuous RL
        self.basic_td3 = BasicTD3(scenario, world, random_state)
        self.commutative_td3 = CommutativeTD3(scenario, world, random_state)
                                        
        # Baselines
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap(scenario, world, random_state)
        self.direct_path = DirectPath(scenario, world, random_state)
                
        self.aerial_agent = Speaker(num_agents, obstacle_radius)
        self.ground_agent = Listener(agent_radius, obstacle_radius)
    
    def retrieve_language(self, approach, problem):            
        approach = getattr(self, approach)
        language = approach.get_language(problem)
            
        return language

    def act(self, problem_instance, language_set, num_episodes):
        approaches = language_set.keys()
        
        rl = language_set['rl']
        voronoi_map = language_set['voronoi_map']
        direct_path = language_set['direct_path']
        approaches = list(language_set.keys())
        direction_set = {approach: None for approach in approaches}

        language_safety = {approach: 0 for approach in approaches}
        ground_agent_success = {approach: 0 for approach in approaches}

        avg_direction_len = {approach: 0 for approach in approaches}
        direction_length = {approach: [] for approach in approaches}
        
        for _ in range(num_episodes):            
            self.env.reset(options={'problem_instance': problem_instance})
            start_state = self.env.state()
            self.aerial_agent.gather_info(start_state)
            
            # Create copy of world to reset to at beginning of each approach
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            direction_set['rl'] = self.aerial_agent.direct(rl)
            direction_set['voronoi_map'] = self.aerial_agent.direct(voronoi_map)
            direction_set['grid_world'] = self.aerial_agent.direct(self.grid_world)
            direction_set['direct_path'] = direct_path
            
            for approach, directions in direction_set.items(): 
                # Penalize if no directions are given
                if None in directions:
                    if approach == 'rl':
                        directions = len(rl)
                    if approach == 'voronoi_map':
                        directions = len(voronoi_map)
                    elif approach == 'grid_world':
                        directions = self.grid_world.graph.number_of_nodes()
                    else:
                        directions = 20
                    direction_length[approach].append(directions)
                    continue
                
                language_safety[approach] += 1
                max_directions = max(len(direction) for direction in directions)
                direction_length[approach].append(max_directions)

                observation, _, termination, truncation, _ = self.env.last()
                while not (termination or truncation):
                    action = self.ground_agent.get_action(observation, directions, approach, language_set[approach])

                    # Epsisode terminates if ground agent doesn't adhere to directions
                    if action is not None:
                        self.env.step(action)
                        observation, _, termination, truncation, _ = self.env.last()
                    else:
                        truncation = True
                                
                    if termination:
                        ground_agent_success[approach] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
                    
                self.env.unwrapped.steps = 0
                self.env.unwrapped.world = copy.deepcopy(backup)
        
        avg_direction_len = {approach: np.mean(direction_length[approach]) for approach in approaches}

        return language_safety, ground_agent_success, avg_direction_len
        

if __name__ == '__main__':
    arguments = get_arguments()
    
    approach = arguments[0]
    problem_instance = arguments[1]
    reward_prediction_type = arguments[2]
    num_agents = arguments[3]
    num_large_obstacles = arguments[4]
    num_small_obstacles = arguments[5]
    random_state = arguments[6]
    render_mode = arguments[7]
    
    ma_cdl = MA_CDL(reward_prediction_type, num_agents, num_large_obstacles, num_small_obstacles, random_state, render_mode)

    language_set = ma_cdl.retrieve_language(approach, problem_instance)
    #     language_safety, ground_agent_success, avg_direction_len = ma_cdl.act(problem_instance, language_set, num_episodes)

    #     all_metrics.append({
    #         'language_safety': language_safety,
    #         'ground_agent_success': ground_agent_success,
    #         'avg_direction_len': avg_direction_len,
    #     })
 
    # plot_metrics(problem_instances, all_metrics, num_episodes)