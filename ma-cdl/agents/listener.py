import numpy as np

from agents.utils.base_aqent import BaseAgent

# TODO: Implemented RL for listener to accept reward

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
        self.actions = np.arange(1,5)
        
    # TODO: Implement Listener's constraints
    def _generate_constraints(self):
        a=3
    
    # TODO: Implement Listener's action selection
    def get_action(self, observation, directions):
        observation = observation[0:2]
        
        obs_region = self.localize(observation)
        goal_region = self.localize(goal)
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            next_region = self.language[label]
            target = next_region.centroid
        
        action = super().get_action(observation, target)
        return action
