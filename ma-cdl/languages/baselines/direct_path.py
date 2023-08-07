from scipy.spatial import cKDTree
from baselines.utils.rrt_star import RRTStar

class DirectPath:
    tree = None
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.planner = RRTStar(
            agent_radius,
            goal_radius,
            obstacle_radius
            )
    
    @staticmethod
    def get_point_index(point):
        return DirectPath.tree.query(point)[1]
    
    # Determine path using RRT*
    def direct(self, start, goal, obstacles):
        direct_path = self.planner.get_path(start, goal, obstacles)
        DirectPath.tree = cKDTree(direct_path)
        return direct_path