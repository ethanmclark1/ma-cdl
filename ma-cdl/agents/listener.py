import numpy as np

from agents.utils.base_aqent import BaseAgent

class Listener(BaseAgent):
    def __init__(self):
        super().__init__()
        self.actions = np.arange(1,5)
        
    # TODO: Implement Listener's constraints
    def _generate_constraints(self):
        a=3
            
    def get_action(self, observation, goal, directions, env):
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
        
        action = super().get_action(observation, target, env)
        return action
