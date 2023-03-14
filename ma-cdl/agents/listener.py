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
        backup = copy.deepcopy(env)
        obs, goal = Point(obs[0:2], goal)
        obs_region = self.localize(obs)
        goal_region = self.localize(goal)
        directions = directions[directions.index(obs_region):]
        
        for action in actions:
            env.step(action)
            obs, _, _, _, _ = env.last()
            obs = obs[0:2]
            obs_point = Point(obs)
            dist = np.linalg.norm(obs - goal) if obs_region == goal_region else inf
            if dist < min_dist:
                min_dist = dist
                optimal_action = action
            env = backup
        
        return optimal_action
            
    
