import copy
import Signal8
import numpy as np

from plotter import plot_metrics
from arguments import get_arguments

from agents.speaker import Speaker
from agents.listener import Listener

from languages.ea import EA
from languages.rl import RL
from languages.bandit import Bandit

from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath


class MA_CDL():
    def __init__(self, num_agents, num_large_obstacles, num_small_obstacles, render_mode, max_cycles=50):
        
        self.env = Signal8.env(
            num_agents=num_agents, 
            num_large_obstacles=num_large_obstacles, 
            num_small_obstacles=num_small_obstacles, 
            render_mode=render_mode,
            max_cycles=max_cycles
            )
        
        scenario = self.env.unwrapped.scenario
        world = self.env.unwrapped.world
        agent_radius = world.agents[0].size
        goal_radius = world.goals[0].size
        obstacle_radius = world.small_obstacles[0].size   
                
        # Context-Dependent Languages
        self.ea = EA(scenario, world)
        self.rl = RL(scenario, world)
        self.bandit = Bandit(scenario, world) 
                
        # Baselines
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap()
        self.direct_path = DirectPath(agent_radius, goal_radius, obstacle_radius)
                
        self.aerial_agent = Speaker(num_agents, obstacle_radius)
        self.ground_agent = [Listener(agent_radius, obstacle_radius) for _ in range(num_agents)]
    
    def retrieve_languages(self, problem_instance):
        approaches = ['ea', 'rl', 'bandit', 'grid_world', 'voronoi_map', 'direct_path']
        language_set = {approach: None for approach in approaches} 
        
        for idx in range(len(approaches)):
            approach = getattr(self, approaches[idx])
            if hasattr(approach, 'get_language'):
                language_set[approaches[idx]] = getattr(self, approaches[idx]).get_language(problem_instance)
             
        return language_set

    def act(self, problem_instance, language_set):
        ea = language_set['ea']
        rl = language_set['rl']
        bandit = language_set['bandit']
        approaches = list(language_set.keys())
        direction_length = {approach: [] for approach in approaches}
    
        language_safety = {approach: 0 for approach in approaches}
        ground_agent_success = {approach: 0 for approach in approaches}
        direction_set = {approach: None for approach in approaches}
        avg_direction_length = {approach: 0 for approach in approaches}
        
        for _ in range(100):            
            self.env.reset(options={'problem_instance': problem_instance})
            start_state = self.env.state()
            self.aerial_agent.gather_info(start_state)
            
            # Create copy of world to reset to at beginning of each approach
            world = self.env.unwrapped.world
            backup = copy.deepcopy(world)
            
            direction_set['ea'] = self.aerial_agent.direct(ea)
            direction_set['rl'] = self.aerial_agent.direct(rl)
            direction_set['bandit'] = self.aerial_agent.direct(bandit)
            direction_set['grid_world'] = self.aerial_agent.direct(self.grid_world)
            direction_set['voronoi_map'] = self.aerial_agent.direct(self.voronoi_map)
            direction_set['direct_path'] = self.aerial_agent.direct(self.direct_path)
            
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
    num_agents, num_large_obstacles, num_small_obstacles, render_mode = get_arguments()
    ma_cdl = MA_CDL(num_agents, num_large_obstacles, num_small_obstacles, render_mode)
        
    problem_instances = ma_cdl.env.unwrapped.world.problem_list
    for problem_instance in problem_instances:
        language_set = ma_cdl.retrieve_languages(problem_instance)
        language_safety, ground_agent_success, avg_direction_length = ma_cdl.act(problem_instance, language_set)
        
        plot_metrics(problem_instance, 
                     language_safety=language_safety,
                     ground_agent_success=ground_agent_success, 
                     avg_direction_length=avg_direction_length
                     )