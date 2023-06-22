from languages.utils.rrt_star import RRTStar

class DirectPath:
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.planner = RRTStar(
            agent_radius,
            goal_radius,
            obstacle_radius
            )
    
    def direct(self, start, goal, obstacles):
        direct_path = self.planner.get_path(start, goal, obstacles)
        return direct_path