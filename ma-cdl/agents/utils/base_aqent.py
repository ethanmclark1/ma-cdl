import torch
import numpy as np

class BaseAgent():
    def __init__(self):
        self.n_actions = np.arange(0,5)
    
    # TODO
    def load(self, problem_instance):
        raise NotImplementedError
    
    # TODO
    def save(self, problem_instance):
        raise NotImplementedError
        
    def localize(self, pos, language):
        try:
            region_idx = list(map(lambda region: region.contains(pos), language)).index(True)
        except:
            region_idx = None
        return region_idx    