import copy
import numpy as np
import matplotlib.pyplot as plt

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language
from agents.utils.baselines.grid_world import GridWorld
from agents.utils.baselines.voronoi_map import VoronoiMap

class MA_CDL2():
    def __init__(self, args):
        self.num_episodes = 100
        self.env = simple_path.env(args)
        self.language = Language(self.env)
        self.speaker = Speaker()
        self.listener = Listener()
        
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap()
    
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
        results, num_actions, direction_set = {}, {}, {}
        # approaches = ['language', 'grid_world', 'voronoi_map']
        # approaches = ['grid_world', 'voronoi_map']
        # approaches = ['voronoi_map']
        approaches = ['grid_world']
        for approach in approaches:
            results[approach] = 0
            num_actions[approach] = []
            direction_set[approach] = None
        
        # self.create_language()
        for _ in range(self.num_episodes):
            self.env.reset()
            start, goal, obstacles = self.env.unwrapped.get_init_conditions()
            
            # direction_set['language'] = self.speaker.direct(start, goal, obstacles)
            direction_set['grid_world'] = self.grid_world.direct(start, goal, obstacles)
            # direction_set['voronoi_map'] = self.voronoi_map.direct(start, goal, obstacles)
            
            self.listener.gather_directions(direction_set)
            
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            for type in direction_set:
                if direction_set[type][1] is None:
                    num_actions[type].append(1e3)
                    break
                                    
                action_count = 0
                self.env.unwrapped.world = copy.deepcopy(backup)
                obs, _, termination, truncation, _ = self.env.last()

                while not (termination or truncation):
                    action = self.listener.get_action(obs, goal, type, self.env)
                    self.env.step(action)
                    obs, _, termination, truncation, _ = self.env.last()
                    action_count += 1
            
                    if termination:
                        results[type] += 1
                        num_actions[type].append(action_count)
                    
        num_actions[type] /= self.num_episodes
        return results, num_actions
    
    def plot(self, results):
        labels = self.env.unwrapped.world.possible_problem_types
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Comparison between Context-Dependent Language, Voronoi Maps, and Grid World on Suite of Problem Scenarios')
        
        num_iterations = len(axes)
        num_labels = len(labels)
        group_size = num_labels // num_iterations
        for i in range(len(axes)):
            start = i * group_size
            end = start + group_size if i < num_iterations - 1 else num_labels
            _labels = labels[start:end]
            _cdl = results['language'][start:end]
            _grid_world = results['grid_world'][start:end]
            _voronoi_map = results['voronoi_map'][start:end]
            
            axes[i].bar(np.arange(len(_labels)) - 0.2, _cdl, width=0.2, label='Context-Dependent Language')
            axes[i].bar(np.arange(len(_labels)), _grid_world, width=0.2, label='Grid World')
            axes[i].bar(np.arange(len(_labels)) + 0.2, _voronoi_map, width=0.2, label='Voronoi Maps')
            axes[i].set_xlabel('Problem Type')
            axes[i].set_ylabel('Success Rate (%)')
            axes[i].set_xticks(np.arange(len(_labels)))
            axes[i].set_xticklabels(_labels)
            axes[i].set_ylim(0, 100)
            axes[i].legend()
        
        plt.savefig('comparison.png')
                        
if __name__ == '__main__':
    args = get_arguments()
    ma_cdl2 = MA_CDL2(args)
    results = ma_cdl2.act()
    ma_cdl2.plot(results)
    