from languages.utils.cdl import CDL

"""Infinitely Armed Bandit"""
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.actions = []
        self.rewards = []
        
        self.action_dims = 3
        