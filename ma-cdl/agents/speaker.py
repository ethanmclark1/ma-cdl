from shapely import Point
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        
    def localize(self, pos, language):
        try:
            region_idx = list(map(lambda region: region.contains(pos), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    # Provide directions for each agent to get to their respective goal
    def direct(self, state, language):
        directions = []
        
        obs_pos = state[-self.num_agents*2:-self.num_agents*2+2]
        obstacles = [Point(obs) for obs in obs_pos]
        
        for idx in range(self.num_agents):
            start_pos = state[idx*2:idx*2+2]
            start_idx = self.localize(Point(start_pos), language)
            goal_pos = state
            goal_idx = self.localize(Point(goal_pos), language)
            
            directions += [a_star(start_idx, goal_idx, obstacles, language)]
                    
        return directions