import copy
import numpy as np

from math import inf
from shapely import Point
from agents.utils.base_aqent import BaseAgent
from agents.utils.baselines.grid_world import GridWorld
from agents.utils.baselines.voronoi_map import VoronoiMap

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
        self.direction_set = None
        self.grid_world = GridWorld()
        self.voronoi_map = VoronoiMap()
        
        
    def gather_directions(self, direction_set):
        self.direction_set = direction_set
    
    # TODO: Implement Listener's constraints
    def get_action(self, observation, goal, type, env):
        min_dist = inf
        actions = np.arange(1, 5)
        
        world = env.unwrapped.world
        backup = copy.deepcopy(world)
        observation = observation[0:2]
        
        directions = self.direction_set[type]
        if type == 'language':
            observation = Point(observation)
            goal =  Point(goal)
            target = self.language(observation, goal, directions)
        elif type == 'grid_world':
            target = self.grid_world(observation, goal)
        elif type == 'voronoi_map':
            target = self.voronoi_map(observation, goal)
        
        target = Point(target)
        # 1: Left, 2: Right, 3: Down, 4: Up
        for action in actions:
            env.step(action)
            observation, _, _, _, _ = env.last()
            observation = Point(observation[0:2])
            dist = observation.distance(target)
            if dist < min_dist:
                min_dist = dist
                optimal_action = action
                
            env.unwrapped.world = copy.deepcopy(backup)
            if env.terminations['agent_0'] or env.truncations['agent_0']:
                env.terminations['agent_0'] = False
                env.truncations['agent_0'] = False
                break
        
        return optimal_action
    
    def language(self, observation, goal, directions):
        obs_region, goal_region = self.localize(observation), self.localize(goal)
        if obs_region == goal_region:
            target = goal
        else:
            a=3
            # label = directions[directions.index(obs_region)+1:][0]
            # region = self.language[label]
            # target = region.centroid
        return target
    
    def grid_world(self, observation, goal):
        directions = self.direction_set['grid_world']
        
        obs_region = self.grid_world.discretize_state(observation)
        goal_region = self.grid_world.discretize_state(goal)
        next_region = directions[directions.index(obs_region)+1:][0]
        target = goal if goal_region == next_region else self.dequantize_state(next_region)
            
        return target
    
    def voronoi_map(self, observation, goal):
        graph, directions, voronoi = self.direction_set['voronoi_map']
        
        target = 3
        return target
            
    
