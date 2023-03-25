from shapely import Point
from agents.utils.search import search
from agents.utils.base_aqent import BaseAgent

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        
    def direct(self, start_pos, goal_pos, obstacles):
        start = Point(start_pos)
        start_idx = self.localize(start)
        goal = Point(goal_pos)
        goal_idx = self.localize(Point(goal_pos))
        obstacles = [Point(obstacle) for obstacle in obstacles]
        directions = search(start_idx, goal_idx, obstacles, self.language, self.name)
        return directions
