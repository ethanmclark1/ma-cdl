import copy

import Signal8
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from agents.speaker import Speaker
from agents.listener import Listener

from languages.td3 import TD3
from languages.bandit import Bandit
from languages.evolutionary_algo import EA

class MA_CDL2():
    def __init__(self, num_agents=1):
        self.env = Signal8.env(num_agents)
        
        agent_radius = self.env.unwrapped.world.agents[0].size
        obstacle_radius = self.env.unwrapped.world.obstacles[1].size
                
        self.ea = EA(agent_radius, obstacle_radius)
        self.td3 = TD3(agent_radius, obstacle_radius)
        self.bandit = Bandit(agent_radius, obstacle_radius) 
        self.speaker = Speaker()
        self.listener = [Listener() for _ in range(num_agents)]

    def act(self):
        approaches = ['ea', 'td3', 'bandit']
        problem_scenarios = Signal8.get_problem_list()
        languages = {approach: None for approach in approaches}
        directions = {approach: None for approach in approaches}
        direction_len = {approach: {scenario: [] for scenario in problem_scenarios} for approach in approaches}
        results = {approach: {scenario: 0 for scenario in problem_scenarios} for approach in approaches}
        
        for _, scenario in product(range(10), problem_scenarios):
            # languages['ea'] = self.ea.get_language(scenario)
            # languages['td3'] = self.td3.get_language(scenario)
            # languages['bandit'] = self.bandit.get_language(scenario)
            
            self.env.reset(options={'problem_name': scenario})
            entities = self.env.unwrapped.get_start_state()
            
            directions['ea'] = self.speaker.direct(entities, languages['ea'])    
            directions['td3'] = self.speaker.direct(entities, languages['td3'])    
            directions['bandit'] = self.speaker.direct(entities, languages['bandit'])
            
            for approach in directions:                
                directions = directions[approach]
                
                direction_len[approach][scenario].append(len(directions))
                observation, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    action = self.listener.get_action(observation, directions)
                    self.env.step(action)
                    observation, reward, termination, truncation, _ = self.env.last()
                                
                    if termination:
                        results[approach][scenario] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
        
        self.env.unwrapped.scenario.stop_scripted_obstacles()
        avg_direction_len = {approach: {scenario: np.mean(values) for scenario, values in scenario_dict.items()} 
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
        num_labels = len(self.problem_scenarios)
        group_size = num_labels // num_iterations
        for i in range(num_iterations):
            start = i * group_size
            end = start + group_size if i < num_iterations - 1 else num_labels
            _labels = self.problem_scenarios[start:end]
            
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
    ma_cdl2 = MA_CDL2(num_agents=2)
    results, avg_direction_len = ma_cdl2.act()
    ma_cdl2.plot(results, avg_direction_len)