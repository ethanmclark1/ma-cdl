import copy
import numpy as np

from math import inf
from shapely import Point
from agents.utils.base_aqent import BaseAgent

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
    
    def get_action(self, obs, goal, directions, env):
        min_dist = inf
        actions = np.arange(1, 5)
        world = env.unwrapped.world
        backup = copy.deepcopy(world)
        obs = Point(obs[0:2])
        goal =  Point(goal)
        obs_region = self.localize(obs)
        goal_region = self.localize(goal)
        
        if obs_region == None or goal_region == None:
            a=3
        
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
            # 1: Left, 2: Right, 3: Down, 4: Up
            if dist <= min_dist:
                min_dist = dist
                optimal_action = action
            env.unwrapped.world = copy.deepcopy(backup)
            
            # Reset termination and truncation flags s.t. the agent can continue
            if env.terminations['agent_0'] or env.truncations['agent_0']:
                env.terminations['agent_0'] = False
                env.truncations['agent_0'] = False
                break
        
        return optimal_action
            
    
