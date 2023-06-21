from shapely import Point
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self, num_agents, obstacle_radius):
        self.agents = []
        self.goals = []
        self.obstacles = None
        self.num_agents = num_agents
        self.obstacle_radius = obstacle_radius
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state):
        obstacle_positions = state[self.num_agents*2 : -self.num_agents*2].reshape(-1, 2)
        self.obstacles = [Point(obstacle_position).buffer(self.obstacle_radius) 
                          for obstacle_position in obstacle_positions]
        
        for idx in range(self.num_agents):
            self.agents += [state[idx*2 : idx*2+2]]
            self.goals += [state[-self.num_agents*2 + idx*2 - 2 : -self.num_agents*2 + idx*2]]
        
    # Find the region that contains the entity
    def localize(self, entity, language):
        try:
            region_idx = list(map(lambda region: region.contains(entity), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    # Provide directions for each agent to get to their respective goal using CDL
    def direct_with_cdl(self, language):
        directions = []
        
        for agent, goal in zip(self.agents, self.goals):
            agent_idx = self.localize(Point(agent), language)
            goal_idx = self.localize(Point(goal), language)
            directions += [a_star(agent_idx, goal_idx, self.obstacles, language)]
                    
        return directions
    
    def direct_with_baseline(self, baseline):
        directions = []
        for agent, goal in zip(self.agents, self.goals):
            directions += [baseline.direct(agent, goal, self.obstacles)]
        
        return directions