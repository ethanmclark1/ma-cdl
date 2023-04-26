import copy
import numpy as np

from math import inf
from scipy.spatial import distance

class BaseAgent():
    def __init__(self):
        self.n_actions = np.arange(0,5)
        
    def localize(self, pos, language):
        try:
            region_idx = list(map(lambda region: region.contains(pos), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    # 0: No-op, 1: Left, 2: Right, 3: Down, 4: Up
    def get_action(self, observation, target, env, region_finder=localize):
        min_dist = inf
        optimal_action = None
        world = env.unwrapped.world
        backup = copy.deepcopy(world)
        obs_region = region_finder(observation)
        target_region = region_finder(target)
        valid_regions = [obs_region, target_region]
        
        for action in self.n_actions:
            env.step(action)
            obs, _, _, _, _ = env.last()
            new_obs = obs[0:2]
            new_obs_region = region_finder(new_obs)
            
            dist = distance.euclidean(target, new_obs)
            if new_obs_region in valid_regions and dist < min_dist:
                min_dist = dist
                optimal_action = action

            env.unwrapped.world = copy.deepcopy(backup)
            if env.terminations['agent_0'] or env.truncations['agent_0']:
                env.terminations['agent_0'] = False
                env.truncations['agent_0'] = False
                optimal_action = action
                break
        
        return optimal_action
                