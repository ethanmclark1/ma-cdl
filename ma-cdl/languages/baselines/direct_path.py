from scipy.spatial import cKDTree
from languages.baselines.utils.rrt_star import RRTStar


class DirectPath:
    tree = None
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.planner = RRTStar(agent_radius, goal_radius, obstacle_radius)
    
    @staticmethod
    def get_point_index(point):
        return DirectPath.tree.query(point)[1]
    
    # Determine path using RRT*
    def direct(self, start, goal, obstacles):
        try:
            direct_path = self.planner.get_path(start, goal, obstacles)
            DirectPath.tree = cKDTree(direct_path)
        except:
            direct_path = None
        return direct_path