import copy
import numpy as np
import blocksworld3d

from languages.rl import RL

from plotter import plot_metrics
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

class MA_CDL():
    def __init__(self, render_mode, max_cycles=50):
        
        self.env = blocksworld3d.env(
            render_mode=render_mode, 
            max_cycles=max_cycles
            )
        
        scenario = self.env.unwrapped.scenario
        world = self.env.unwrapped.world
        agent_radius = world.agents[0].radius
        obstacle_radius = world.small_obstacles[0].radius 
        
        # Context-Dependent Language
        self.rl = RL(scenario, world)
                
        self.aerial_agent = Speaker(obstacle_radius)
        self.ground_agent = Listener(agent_radius, obstacle_radius)
    
    def retrieve_language(self, problem_instance):
        return self.rl.get_language(problem_instance)
             
    def act(self, problem_instance, language_set, num_episodes):
        rl = language_set['rl']
        approaches = list(language_set.keys())
        direction_length = {approach: [] for approach in approaches}
    
        language_safety = {approach: 0 for approach in approaches}
        ground_agent_success = {approach: 0 for approach in approaches}
        direction_set = {approach: None for approach in approaches}
        avg_direction_length = {approach: 0 for approach in approaches}
        
        for _ in range(num_episodes):            
            self.env.reset(options={'problem_instance': problem_instance})
            start_state = self.env.state()
            self.aerial_agent.gather_info(start_state)
            
            # Create copy of world to reset to at beginning of each approach
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            direction_set['rl'] = self.aerial_agent.direct(rl)
            
            for approach, directions in direction_set.items(): 
                if None in directions:
                    continue
                
                language_safety[approach] += 1
                max_directions = max(len(direction) for direction in directions)
                direction_length[approach].append(max_directions)

                i = 0 
                observation, _, termination, truncation, _ = self.env.last()
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
        
        avg_direction_length = {approach: np.mean(direction_length[approach]) for approach in approaches}

        return language_safety, ground_agent_success, avg_direction_length
        

if __name__ == '__main__':
    render_mode = get_arguments()
    ma_cdl = MA_CDL(render_mode)

    num_episodes = 10000
    problem_instances = ma_cdl.env.unwrapped.world.problem_list
    all_metrics = []
    for problem_instance in problem_instances:
        language_set = ma_cdl.retrieve_languages(problem_instance)
        language_safety, ground_agent_success, avg_direction_length = ma_cdl.act(problem_instance, language_set, num_episodes)

        all_metrics.append({
            'language_safety': language_safety,
            'ground_agent_success': ground_agent_success,
            'avg_direction_length': avg_direction_length
        })
 
    plot_metrics(problem_instances, all_metrics, num_episodes)