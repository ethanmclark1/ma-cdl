import copy
import numpy as np

from math import inf
from shapely import points as Point
from agents.base_aqent import BaseAgent

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
    
    def get_action(self, obs, goal, directions, env):
        min_dist = inf
        actions = np.arange(0, 5)
        world = env.unwrapped.world
        backup = copy.deepcopy(world)
        obs, goal = Point(obs[0:2], goal)
        obs_region = self.localize(obs)
        goal_region = self.localize(goal)
        
        if obs_region == goal_region:
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            region = self.language[label]
            target = region.centroid
        
        for action in actions:
            env.step(action)
            obs, _, _, _, _ = env.last()
            obs = Point(obs[0:2])
            dist = obs.distance(target)
            if dist <= min_dist:
                min_dist = dist
                optimal_action = action
            env.unwrapped.world = copy.deepcopy(backup)
        
        return optimal_action
            
    
