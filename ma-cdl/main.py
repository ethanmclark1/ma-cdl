import Signal8
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.td3 import TD3
from languages.bandit import Bandit
from languages.evolutionary_algo import EA

class MA_CDL():
    def __init__(self, problem_type, num_agents, render_mode):
        self.env = Signal8.env(problem_type, num_agents, render_mode)
        
        agent_radius = self.env.unwrapped.world.agents[0].size
        obstacle_radius = self.env.unwrapped.world.landmarks[0].size
        obs_dim = self.env.observation_space(self.env.possible_agents[0]).shape[0]
                
        self.ea = EA(agent_radius, obstacle_radius)
        self.td3 = TD3(agent_radius, obstacle_radius)
        self.bandit = Bandit(agent_radius, obstacle_radius) 
        self.speaker = Speaker()
        self.listener = [Listener(obs_dim) for _ in range(num_agents)]
        
    def get_languages(self, problem_type):
        approaches = ['ea', 'td3', 'bandit']
        problem_instances = [problem_type + f'_{i}' for i in range(4)]
        language_set = {approach: {instance: None for instance in problem_instances} for approach in approaches}   
        
        world = self.env.unwrapped.world
        scenario = self.env.unwrapped.scenario
        for approach, instance in zip(approaches, problem_instances):
            language_set[approach][instance] = self.ea.get_language(instance, scenario, world)
            language_set[approach][instance] = self.td3.get_language(instance, scenario, world)
            language_set[approach][instance] = self.bandit.get_language(instance, scenario, world)     
        
        return language_set

    def act(self, problem_type, language_set):
        approaches = list(language_set.keys())
        direction_set = {approach: None for approach in approaches}
        problem_instances = [problem_type + f'_{i}' for i in range(4)]
        direction_len = {approach: {instance: [] for instance in problem_instances} for approach in approaches}
        results = {approach: {instance: 0 for instance in problem_instances} for approach in approaches}
        
        for _, instance in product(range(10), problem_instances):
            ea = language_set['ea'][instance]
            td3 = language_set['td3'][instance]
            bandit = language_set['bandit'][instance]
            
            instance_num = int(instance[-1])
            self.env.reset(options={'instance_num': instance_num})
            entity_positions = self.env.unwrapped.get_start_state()
            
            direction_set['ea'] = self.speaker.direct(entity_positions, ea)    
            direction_set['td3'] = self.speaker.direct(entity_positions, td3)    
            direction_set['bandit'] = self.speaker.direct(entity_positions, bandit)
            
            # TODO: Implement functionality for multiple listening agents
            for approach in direction_set:                
                directions = direction_set[approach]
                direction_len[approach][instance].append(len(directions))
                observation, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    action = self.listener.get_action(observation, directions)
                    self.env.step(action)
                    observation, _, termination, truncation, _ = self.env.last()
                    reward = self.speaker.give_reward(observation, directions, termination, truncation)
                                
                    if termination:
                        results[approach][instance] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
                    
                    # TODO: Figure out how to reward speaker and listener
        
        self.env.close()
        avg_direction_len = {approach: {instance: np.mean(values) for instance, values in scenario_dict.items()} 
                         for approach, scenario_dict in direction_len.items()}
        return results, avg_direction_len
    
    def plot(self, results, avg_direction_len):        
        # Create the first figure for success rates
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle('Success Rate Comparison between Context-Dependent Language, Voronoi Maps, and Grid World on Suite of Problem Scenarios')
        
        # Create the second figure for average direction lengths
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle('Average Direction Length Comparison between Context-Dependent Language, Voronoi Maps, and Grid World on Suite of Problem Scenarios')
        
        num_iterations = len(axes1)
        num_labels = len(self.problem_instances)
        group_size = num_labels // num_iterations
        for i in range(num_iterations):
            start = i * group_size
            end = start + group_size if i < num_iterations - 1 else num_labels
            _labels = self.problem_instances[start:end]
            
            _cdl_results = list(results['language'].values())[start:end]
            _grid_world_results = list(results['grid_world'].values())[start:end]
            _voronoi_map_results = list(results['voronoi_map'].values())[start:end]

            # Plot the success rate graphs
            axes1[i].bar(np.arange(len(_labels)) - 0.2, _cdl_results, width=0.2, label='Context-Dependent Language')
            axes1[i].bar(np.arange(len(_labels)), _grid_world_results, width=0.2, label='Grid World')
            axes1[i].bar(np.arange(len(_labels)) + 0.2, _voronoi_map_results, width=0.2, label='Voronoi Maps')
            axes1[i].set_xlabel('Problem Type')
            axes1[i].set_ylabel('Success Rate (%)')
            axes1[i].set_xticks(np.arange(len(_labels)))
            axes1[i].set_xticklabels(_labels)
            axes1[i].set_ylim(0, 100)
            axes1[i].legend()

            _cdl_direction_len = list(avg_direction_len['language'].values())[start:end]
            _grid_world_direction_len = list(avg_direction_len['grid_world'].values())[start:end]
            _voronoi_map_direction_len = list(avg_direction_len['voronoi_map'].values())[start:end]
            
            # Plot the average direction length graphs
            axes2[i].bar(np.arange(len(_labels)) - 0.2, _cdl_direction_len, width=0.2, label='Context-Dependent Language')
            axes2[i].bar(np.arange(len(_labels)), _grid_world_direction_len, width=0.2, label='Grid World')
            axes2[i].bar(np.arange(len(_labels)) + 0.2, _voronoi_map_direction_len, width=0.2, label='Voronoi Maps')
            axes2[i].set_xlabel('Problem Type')
            axes2[i].set_ylabel('Average Direction Length')
            axes2[i].set_xticks(np.arange(len(_labels)))
            axes2[i].set_xticklabels(_labels)
            axes2[i].legend()
        
        # Save the figures
        fig1.savefig('success_rates.png')
        fig2.savefig('average_direction_lengths.png')
        
        # Show the plots in separate windows
        plt.show()

if __name__ == '__main__':
    problem_type, num_agents, render_mode = get_arguments()
    ma_cdl = MA_CDL(problem_type, num_agents, render_mode)
    
    language_set = ma_cdl.get_languages(problem_type)
    results, avg_direction_len = ma_cdl.act(problem_type, language_set)
    ma_cdl.plot(results, avg_direction_len)