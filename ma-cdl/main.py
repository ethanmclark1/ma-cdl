import copy
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language
from agents.utils.baselines.grid_world import GridWorld
from agents.utils.baselines.voronoi_map import VoronoiMap

class MA_CDL2():
    def __init__(self, args):
        self.num_episodes = 1
        self.env = simple_path.env(args)
        self.language = Language(self.env)
        self.speaker = Speaker()
        self.listener = Listener()
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap()
        self.problem_scenarios = self.env.unwrapped.world.possible_problem_scenarios
    
    # Passes language to both speaker and listener
    def create_language(self):
        try:
            language = self.language.load()
        except:
            print(f'No existing language found for "{self.env.unwrapped.world.problem_type}" problem type.')
            print('Creating new language...')
            
        language = self.language.create()
        self.speaker.set_language(language)
        self.listener.set_language(language)

    def act(self):
        # approaches = ['language', 'grid_world', 'voronoi_map']
        # approaches = ['grid_world', 'voronoi_map']
        approaches = ['grid_world']
        direction_set = {approach: None for approach in approaches}
        direction_len = {approach: {scenario: [] for scenario in self.problem_scenarios} for approach in approaches}
        results = {approach: {scenario: 0 for scenario in self.problem_scenarios} for approach in approaches}
        
        # self.create_language()
        for _, scenario in product(range(self.num_episodes), self.problem_scenarios):
            self.env.reset(options={'problem_name': scenario})
            start, goal, obstacles = self.env.unwrapped.get_init_conditions()
            
            # direction_set['language'] = self.speaker.direct(start, goal, obstacles)
            direction_set['grid_world'] = self.grid_world.direct(start, goal, obstacles)
            # direction_set['voronoi_map'] = self.voronoi_map.direct(start, goal, obstacles)
                        
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            for approach in direction_set:                
                directions = direction_set[approach]
                if directions is None:
                    continue
                
                direction_len[approach][scenario].append(len(directions))
                self.env.unwrapped.world = copy.deepcopy(backup)
                obs, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    if approach == 'listener':
                        action = self.listener.get_action(obs, goal, directions, self.env)
                    elif approach == 'grid_world':
                        action = self.grid_world.get_action(obs, goal, directions, self.env)
                    elif approach == 'voronoi_map':
                        action = self.voronoi_map.get_action(obs, goal, directions, self.env)
                        
                    self.env.step(action)
                    obs, _, termination, truncation, _ = self.env.last()
                                
                    if termination:
                        results[approach][scenario] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
        
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
            
            _cdl_results = results['language'][start:end]
            _cdl_direction_len = avg_direction_len['language'][start:end]
            _grid_world_results = results['grid_world'][start:end]
            _grid_world_direction_len = avg_direction_len['grid_world'][start:end]
            _voronoi_map_results = results['voronoi_map'][start:end]
            _voronoi_map_direction_len = avg_direction_len['voronoi_map'][start:end]

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
    ma_cdl2 = MA_CDL2(args)
    results, avg_direction_len = ma_cdl2.act()
    ma_cdl2.plot(results, avg_direction_len)