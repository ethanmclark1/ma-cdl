from shapely import Point
from agents.utils.a_star import a_star
from agents.utils.base_aqent import BaseAgent

# TODO: Implemented RL for speaker to accept reward based on listeners adherence

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
    
    # TODO: Figure out way to account for dynamic obstacles
    def direct(self, entity_positions, language):
        directions = []
        
        agent_pos = entity_positions['agents']
        goal_pos = entity_positions['goals']
        obs_pos = entity_positions['obstacles']
        obstacles = [Point(obs) for obs in obs_pos]
        
        for a_pos, g_pos in zip(agent_pos, goal_pos):
            start_idx = self.localize(Point(a_pos), language)
            goal_idx = self.localize(Point(g_pos), language)
            directions += [a_star(start_idx, goal_idx, obstacles, language)]
        return directions
    
    # TODO: Give reward to listener based on directions
    def give_reward(self, observation, directions, termination, truncation):
        return 0.0