import copy
import numpy as np

from math import inf
from shapely import Point

class BaseAgent():
    def __init__(self):
        self.language = None
        self.n_actions = np.arange(1,5)
        
    def set_language(self, language):
        self.language = language
    
    def localize(self, pos):
        try:
            region_idx = list(map(lambda region: region.contains(pos), self.language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    # 1: Left, 2: Right, 3: Down, 4: Up
    def get_action(self, observation, target, env):
        min_dist = inf
        world = env.unwrapped.world
        backup = copy.deepcopy(world)
        target = Point(target)
        
        for action in self.n_actions:
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
                optimal_action = action
                break
        
        return optimal_action
                