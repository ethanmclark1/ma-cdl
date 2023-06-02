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
        self.env = Signal8.env(render_mode=None)
        
        num_agents = len(self.env.unwrapped.world.agents)
        agent_radius = self.env.unwrapped.world.agents[0].size
        obstacle_radius = self.env.unwrapped.world.obstacles[1].size
                
        self.ea = EA(agent_radius, obstacle_radius)
        self.td3 = TD3(agent_radius, obstacle_radius)
        self.bandit = Bandit(agent_radius, obstacle_radius) 
        self.speaker = Speaker()
        self.listener = [Listener() for _ in range(num_agents)]
        
    def get_languages(self, problem_scenarios):
        approaches = ['ea', 'td3', 'bandit']
        language_set = {approach: {scenario: None for scenario in problem_scenarios} for approach in approaches}   
        
        world = self.env.unwrapped.world
        for approach, scenario in zip(approaches, problem_scenarios):
            language_set[approach][scenario] = self.ea.get_language(scenario, world)
            language_set[approach][scenario] = self.td3.get_language(scenario, world)
            language_set[approach][scenario] = self.bandit.get_language(scenario, world)     
        
        return language_set

    def act(self, problem_scenarios, language_set):
        approaches = list(language_set.keys())
        direction_set = {approach: None for approach in approaches}
        direction_len = {approach: {scenario: [] for scenario in self.problem_scenarios} for approach in approaches}
        results = {approach: {scenario: 0 for scenario in self.problem_scenarios} for approach in approaches}
        
        for _, scenario in product(range(10), problem_scenarios):
            language_set['ea'] = self.ea.get_language(scenario)
            language_set['td3'] = self.td3.get_language(scenario)
            language_set['bandit'] = self.bandit.get_language(scenario)
            
            self.env.reset(options={'problem_name': scenario})
            entity_positions = self.env.unwrapped.get_start_state()
            
            direction_set['ea'] = self.speaker.direct(entity_positions, language_set['ea'])    
            direction_set['td3'] = self.speaker.direct(entity_positions, language_set['td3'])    
            direction_set['bandit'] = self.speaker.direct(entity_positions, language_set['bandit'])
            
            # TODO: Implement functionality for multiple listening agents
            for approach in direction_set:                
                directions = direction_set[approach]
                direction_len[approach][scenario].append(len(directions))
                observation, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    action = self.listener.get_action(observation, directions)
                    self.env.step(action)
                    observation, _, termination, truncation, _ = self.env.last()
                    reward = self.speaker.give_reward(observation, directions, termination, truncation)
                                
                    if termination:
                        results[approach][scenario] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
                    
                    # TODO: Figure out how to reward speaker and listener
        
        self.env.close()
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
    args = get_arguments()
    ma_cdl = MA_CDL(args)
    
    problems = Signal8.get_problem_list()
    disaster = [problem for problem in problems if problem.startswith('disaster_response')]
    farming = [problem for problem in problems if problem.startswith('precision_farming')]
    
    language_set = ma_cdl.get_languages(disaster)
    results, avg_direction_len = ma_cdl.act(language_set)
    ma_cdl.plot(results, avg_direction_len)