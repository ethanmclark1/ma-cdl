import numpy as np
from agents.utils.base_agent import BaseAgent
from agents.utils.potential_field import PathPlanner

class Listener(BaseAgent):
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        super().__init__(agent_radius, goal_radius, obstacle_radius)
        self.language = None
        self.planner = PathPlanner(agent_radius, goal_radius, obstacle_radius)
        
    # Convert discrete state to continous state for GridWorld
    def dequantize_state(self, discretized_state):
        continuous_state = []
        
        for i, val in enumerate(discretized_state):
            bin_width = (self.state_ranges[i][1] - self.state_ranges[i][0]) / self.n_bins[i]
            continuous_val = self.state_ranges[i][0] + (val + 0.5) * bin_width
            continuous_state.append(continuous_val)
        return tuple(continuous_state)
                
    """
    1. Gather position and velocity of agent
    2. Determine where agent is in the environment
    3. Determine where agent needs to go to
    4. Path Plan from current position to desired position, generating only one action
    """
    def get_action(self, observation, directions, approach):
        agent_pos = observation[:2]
        agent_vel = observation[2:4]        
        obs_pos = observation[4:-2].reshape(-1, 2)
        goal_pos = observation[-2:]
        
        if approach in self.languages:
            obs_region = self.localize(agent_pos)
            goal_region = self.localize(goal_pos)
        
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            next_region = self.language[label]
            target = next_region.centroid
        
        action = 0
        
        return action