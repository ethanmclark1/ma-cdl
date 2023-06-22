import numpy as np
from agents.utils.potential_field import PathPlanner

class Listener:
    def __init__(self, agent_radius, goal_radius, obs_radius):
        self.language = None
        self.planner = PathPlanner(agent_radius, goal_radius, obs_radius)
        
        
    # TODO: Implement this
    """
    1. Gather position and velocity of agent
    2. Determine where agent is in the environment
    3. Determine where agent needs to go to
    4. Path Plan from current position to desired position, generating only one action
    """
    def get_action(self, observation, directions):
        agent_pos = observation[:2]
        agent_vel = observation[2:4]
        
        agent_state = observation[:4]
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
        
        action = 0
        
        return action