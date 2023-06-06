import os
import torch
import numpy as np

from agents.utils.networks import ListenerNetwork

class Listener:
    def __init__(self, problem_type, obs_dim):
        self.language = None
        self.listener_network = ListenerNetwork(obs_dim)
    
    # TODO: Implement Listener's constraints
    def _generate_constraints(self):
        a=3
    
    # TODO: Implement Listener's action selection considering own constraints on top of directions
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
    
    # Reward speaker based on distance from obstacles and nearness to goal
    def reward_to_speaker(self, observation):
        speaker_reward = 0.0
        for relative_pos in observation[2:-2]:
            speaker_reward += np.linalg.norm(relative_pos)
        speaker_reward += 1.1*np.linalg.norm(observation[-2:])
        
        return speaker_reward
