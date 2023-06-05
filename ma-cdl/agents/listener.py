import os
import torch

from agents.utils.base_aqent import BaseAgent
from agents.utils.networks import ListenerNetwork

class Listener(BaseAgent):
    def __init__(self, problem_instance, obs_dim):
        super().__init__()
        self.listener_network = ListenerNetwork(obs_dim)
    
        
    # TODO: Implement Listener's constraints
    def _generate_constraints(self):
        a=3
    
    # TODO: Implement Listener's action selection
    def get_action(self, observation, directions):
        agent_pos = observation[:2]
        goal_pos = observation[-2:]
        obs_region = self.localize(agent_pos)
        goal_region = self.localize(goal_pos)
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            next_region = self.language[label]
            target = next_region.centroid
        
        action = super().get_action(observation, target)
        return action
