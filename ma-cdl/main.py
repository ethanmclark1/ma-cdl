import copy
import numpy as np
import gymnasium as gym

from plotter import plot_metrics

from agents.speaker import Speaker
from agents.listener import Listener

from languages.discrete import Discrete

class MA_CDL():
    def __init__(self, natural=False, sab=False):
        self.env = gym.make('Blackjack-v1', natural=natural, sab=sab)
    
        # Context-Dependent Language
        self.rl = Discrete()
                
        self.observer = Speaker()
        self.player = Listener()

    def act(self, num_episodes):
        language = self.rl.get_language()
        self.observer.set_language(language)
        
        for _ in range(num_episodes):        
            obs, _ = self.env.reset()
            sum, dealer, has_ace = obs
            self.observer.set_state(dealer)
            self.player.set_state(sum, has_ace)
        
            self.observer.direct(language)
            termination = truncation = False
            
            while not (termination or truncation):
                action = self.ground_agent[i].get_action(observation, directions[i], approach, language_set[approach])

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
                
                i += 1
                i %= len(directions)
                    
                self.env.unwrapped.steps = 0
                self.env.unwrapped.world = copy.deepcopy(backup)
        
        avg_direction_len = {approach: np.mean(direction_length[approach]) for approach in approaches}

        return language_safety, ground_agent_success, avg_direction_len
        

if __name__ == '__main__':
    ma_cdl = MA_CDL()

    all_metrics = []
    num_episodes = 10000
    ma_cdl.act(num_episodes)
    # for problem_instance in problem_instances:
    #     language_set = ma_cdl.retrieve_languages(problem_instance)
    #     language_safety, ground_agent_success, avg_direction_len = ma_cdl.act(problem_instance, language_set, num_episodes)

    #     all_metrics.append({
    #         'language_safety': language_safety,
    #         'ground_agent_success': ground_agent_success,
    #         'avg_direction_len': avg_direction_len,
    #     })
 
    # plot_metrics(problem_instances, all_metrics, num_episodes)