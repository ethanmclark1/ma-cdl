import numpy as np

from shapely import points
from agents.base_aqent import BaseAgent

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
    
    def get_action(self, obs, goal, directions):
        obs, goal = points(obs[0:2], goal)
        obs_region = self.localize(obs)
        goal_region = self.localize(goal)
        # Minimize distance to goal
        if obs_region == goal_region:
            a=3
        # Minimize distance to next region
        else:
            a=3
            
    
