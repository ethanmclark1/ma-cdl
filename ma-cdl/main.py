import copy
import Signal8
import numpy as np
import matplotlib.pyplot as plt

from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.td3 import TD3
from languages.bandit import Bandit
from languages.evolutionary_algo import EA

from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath

class MA_CDL():
    def __init__(self, num_agents, num_large_obstacles, num_small_obstacles, render_mode):
        
        self.env = Signal8.env(
            num_agents, 
            num_large_obstacles, 
            num_small_obstacles, 
            render_mode
            )
        
        scenario = self.env.unwrapped.scenario
        world = self.env.unwrapped.world
        agent_radius = world.agents[0].size
        goal_radius = world.goals[0].size
        obs_radius = world.small_obstacles[0].size
        
        # Context-Dependent Languages
        self.ea = EA(scenario, world)
        self.td3 = TD3(scenario, world)
        self.bandit = Bandit(scenario, world) 
        
        # Baselines
        self.grid_world = GridWorld(agent_radius, goal_radius, obs_radius)
        self.voronoi_map = VoronoiMap(agent_radius, goal_radius, obs_radius)
        self.direct_path = DirectPath(agent_radius, goal_radius, obs_radius)
                
        self.speaker = Speaker(num_agents, obs_radius)
        self.listener = [Listener(agent_radius, goal_radius, obs_radius) for _ in range(num_agents)]
    
    def retrieve_languages(self, problem_instance):
        # approaches = ['ea', 'td3', 'bandit', 'grid_world', 'voronoi_map', 'direct_path']
        approaches = ['direct_path']
        language_set = {approach: None for approach in approaches} 
        
        for idx in range(len(approaches)):
            approach = getattr(self, approaches[idx])
            if hasattr(approach, 'get_language'):
                language_set[approach] = getattr(self, approach).get_language(problem_instance)
             
        return language_set

    def act(self, problem_instance, language_set):
        # ea = language_set['ea']
        # td3 = language_set['td3']
        # bandit = language_set['bandit']
        approaches = list(language_set.keys())
    
        results = {approach: 0 for approach in approaches}
        direction_len = {approach: [] for approach in approaches}
        direction_set = {approach: None for approach in approaches}
        
        for _ in range(10):            
            self.env.reset(options={'problem_instance': problem_instance})
            start_state = self.env.state()
            self.speaker.gather_info(start_state)
            
            # Create copy of world to reset to for each approach
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            # direction_set['ea'] = self.speaker.direct(ea)
            # direction_set['td3'] = self.speaker.direct(td3)
            # direction_set['bandit'] = self.speaker.direct(bandit)
            # direction_set['grid_world'] = self.speaker.direct(self.grid_world)
            direction_set['voronoi_map'] = self.speaker.direct(self.voronoi_map)
            direction_set['direct_path'] = self.speaker.direct(self.direct_path)
            
            for approach in direction_set.keys():  
                i = 0 
                directions = direction_set[approach]
                max_directions = max(len(direction) for direction in directions)
                direction_len[approach].append(max_directions)
                observation, _, termination, truncation, _ = self.env.last()
                
                while not (termination or truncation):
                    action = self.listener[i].get_action(observation, directions[i])
                    self.env.step(action)
                    observation, _, termination, truncation, _ = self.env.last()
                                
                    if termination:
                        results[approach] += 1
                        self.env.terminations['agent_0'] = False
                        break
                    elif truncation:
                        self.env.truncations['agent_0'] = False
                        break   
                    
                    i += 1
                    i %= len(directions)
                    
                self.env.unwrapped.world = copy.deepcopy(backup)
                            
        self.env.close()
        avg_direction_len = {approach: {instance: np.mean(values) for instance, values in scenario_dict.items()} 
                         for approach, scenario_dict in direction_len.items()}
        return results, avg_direction_len
    
    def plot(self, problem_instance, results, avg_direction_len):        
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle(f'Success Rate Comparison between Context-Dependent Languages, Voronoi Map, Grid World, and Direct Path on {problem_instance.capitalize()} Problem Instance')
        
        num_approaches = len(results.keys())        
            
        ea_results = list(results['ea'].values())
        td3_results = list(results['td3'].values())
        bandit_results = list(results['bandit'].values())
        grid_world_results = list(results['grid_world'].values())
        voronoi_map_results = list(results['voronoi_map'].values())
        direct_path_results = list(results['direct_path'].values())

        # Plot the success rate graphs
        axes1.bar(np.arange(len(num_approaches)) - 0.6, ea_results, width=0.2, label='Evolutionary Algorithm')
        axes1.bar(np.arange(len(num_approaches)) - 0.4, td3_results, width=0.2, label='TD3')
        axes1.bar(np.arange(len(num_approaches)) - 0.2, bandit_results, width=0.2, label='Bandit')
        axes1.bar(np.arange(len(num_approaches)) + 0.2, grid_world_results, width=0.2, label='Grid World')
        axes1.bar(np.arange(len(num_approaches)) + 0.4, voronoi_map_results, width=0.2, label='Voronoi Maps')
        axes1.bar(np.arange(len(num_approaches)) + 0.6, direct_path_results, width=0.2, label='Direct Path')
        axes1.set_xlabel('Problem Type')
        axes1.set_ylabel('Success Rate (%)')
        axes1.set_xticks(np.arange(len(num_approaches)))
        axes1.set_xticklabels(num_approaches)
        axes1.set_ylim(0, 100)
        axes1.legend()
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle(f'Average Direction Length Comparison between Context-Dependent Languages, Voronoi Map, Grid World, and Direct Path on {problem_instance.capitalize()} Problem Instance')
        
        ea_direction_len = list(avg_direction_len['ea'].values())
        td3_direction_len = list(avg_direction_len['td3'].values())
        bandit_direction_len = list(avg_direction_len['bandit'].values())
        grid_world_direction_len = list(avg_direction_len['grid_world'].values())
        voronoi_map_direction_len = list(avg_direction_len['voronoi_map'].values())
        direct_path_direction_len = list(avg_direction_len['direct_path'].values())
        
        # Plot the average direction length graphs
        axes2.bar(np.arange(len(num_approaches)) - 0.6, ea_direction_len, width=0.2, label='Evolutionary Algorithm')
        axes2.bar(np.arange(len(num_approaches)) - 0.4, td3_direction_len, width=0.2, label='TD3')
        axes2.bar(np.arange(len(num_approaches)) - 0.2, bandit_direction_len, width=0.2, label='Bandit')
        axes2.bar(np.arange(len(num_approaches)) + 0.2, grid_world_direction_len, width=0.2, label='Grid World')
        axes2.bar(np.arange(len(num_approaches)) + 0.4, voronoi_map_direction_len, width=0.2, label='Voronoi Maps')
        axes2.bar(np.arange(len(num_approaches)) + 0.6, direct_path_direction_len, width=0.2, label='Direct Path')
        axes2.set_xlabel('Problem Type')
        axes2.set_ylabel('Average Direction Length')
        axes2.set_xticks(np.arange(len(num_approaches)))
        axes2.set_xticklabels(num_approaches)
        axes2.legend()
        
        fig1.savefig(f'results/{problem_instance}/success_rates.png')
        fig2.savefig(f'results/{problem_instance}/average_direction_lengths.png')        
        plt.show()
        

if __name__ == '__main__':
    num_agents, num_large_obstacles, num_small_obstacles, render_mode = get_arguments()
    
    ma_cdl = MA_CDL(
        num_agents, 
        num_large_obstacles, 
        num_small_obstacles, 
        render_mode
        )
        
    problem_instances = ma_cdl.env.unwrapped.world.problem_list
    for problem_instance in problem_instances:
        language_set = ma_cdl.retrieve_languages(problem_instance)
        results, avg_direction_len = ma_cdl.act(problem_instance, language_set)
        ma_cdl.plot(problem_instance, results, avg_direction_len)