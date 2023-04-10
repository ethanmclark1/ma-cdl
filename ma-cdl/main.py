import copy
import numpy as np
import matplotlib.pyplot as plt

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language

class MA_CDL2():
    def __init__(self, args):
        self.num_episodes = 100
        self.env = simple_path.env(args)
        self.language = Language(self.env)
        self.speaker = Speaker()
        self.listener = Listener()
    
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
        success = 0
        self.create_language()
        
        for episode in range(self.self.num_episodes):
            start, goal, obstacles = self.env.unwrapped.get_init_conditions()
            directions = self.speaker.direct(start, goal, obstacles)
            obs, _, termination, truncation, _ = self.env.last()
            
            while not (termination or truncation):
                action = self.listener.get_action(obs, goal, directions, self.env)
                self.env.step(action)
                obs, _, termination, truncation, _ = self.env.last()
            
            if termination:
                success += 1
                print('Mission success!')
            else:
                print('Mission failed!')
        
        return success
    
    def plot(self, cdl, baseline):
        labels = self.env.unwrapped.world.possible_problem_types
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Comparison between Context-Dependent Language and Baseline (Voroni Maps) on Suite of Problem Scenarios')
        
        num_iterations = len(axes)
        num_labels = len(labels)
        group_size = num_labels // num_iterations
        for i in range(len(axes)):
            start = i * group_size
            end = start + group_size if i < num_iterations - 1 else num_labels
            _labels = labels[start:end]
            _cdl = cdl[start:end]
            _baseline = baseline[start:end]
            
            axes[i].bar(np.arange(len(_labels)) - 0.2, _cdl, width=0.4, label='Context-Dependent Language')
            axes[i].bar(np.arange(len(_labels)) + 0.2, _baseline, width=0.4, label='Baseline')
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
    successes, failures = ma_cdl2.act()
    ma_cdl2.plot(successes, failures)
    