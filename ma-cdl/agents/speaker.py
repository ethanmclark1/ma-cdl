from shapely import Point
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self, num_agents, agent_radius, goal_radius, obstacle_radius):
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
    
    # TODO: Account for agent size
    def localize(self, pos, language):
        try:
            region_idx = list(map(lambda region: region.contains(pos), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    # Provide directions for each agent to get to their respective goal
    def direct(self, state, language):
        directions = []
        
        obstacle_positions = state[self.num_agents*2 : -self.num_agents*2].reshape(-1, 2)
        obstacles = [Point(obstacle_position).buffer(self.obstacle_radius) 
                     for obstacle_position in obstacle_positions]
        
        for idx in range(self.num_agents):
            agent = Point(state[idx*2 : idx*2+2]).buffer(self.agent_radius)
            agent_idx = self.localize(agent, language)
            goal = Point(state[-self.num_agents*2+idx : -self.num_agents*2+idx+2]).buffer(self.goal_radius)
            goal_idx = self.localize(goal, language)
            
            directions += [a_star(agent_idx, goal_idx, obstacles, language)]
                    
        return directions