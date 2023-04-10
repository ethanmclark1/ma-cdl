from shapely import Point
from agents.utils.base_aqent import BaseAgent
from agents.utils.path_finder.a_star import a_star

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def direct(self, start_pos, goal_pos, obstacles):
        start = Point(start_pos)
        start_idx = self.localize(start)
        goal = Point(goal_pos)
        goal_idx = self.localize(Point(goal_pos))
        obstacles = [Point(obstacle) for obstacle in obstacles]
        directions = a_star(start_idx, goal_idx, obstacles, self.language)
        return directions
    
    """"
    Baselines
        1. Discretize environment into GridWorld
        2. Voronoi Maps
        3. 
    """
