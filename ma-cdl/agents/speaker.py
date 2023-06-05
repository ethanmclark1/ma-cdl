from shapely import Point
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self):
        self.language = None
        
    def localize(self, pos, language):
        try:
            region_idx = list(map(lambda region: region.contains(pos), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
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
    
    # Give reward to listener based on adherence to directions
    def reward_to_listener(self, observation, directions, termination, truncation):
        a=3