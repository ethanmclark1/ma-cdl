from shapely import Point
from agents.utils.a_star import a_star
from agents.utils.base_aqent import BaseAgent

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def direct(self, entities, language):
        agent_pos = entities['agents']
        goal_pos = entities['goals']
        obs_pos = entities['obstacles']
        start_idx = self.localize(Point(start_pos), language)
        goal_idx = self.localize(Point(goal_pos), language)
        obstacles = [Point(obs) for obs in obstacles]
        directions = a_star(start_idx, goal_idx, obstacles, language)
        return directions
