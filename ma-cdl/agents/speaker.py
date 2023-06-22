from shapely import Point
from agents.utils.a_star import a_star

class Speaker:
    def __init__(self, num_agents, obstacle_radius):
        self.agents = []
        self.goals = []
        self.obstacles = None
        self.num_agents = num_agents
        self.obstacle_radius = obstacle_radius
        self.languages = ['EA', 'TD3', 'Bandit']
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state):
        
        for idx in range(self.num_agents):
            self.agents += [state[idx*2 : idx*2+2]]
            self.goals += [state[self.num_agents*2 + idx*2 : self.num_agents*2 + idx*2 + 2]]
            
        self.obstacles = state[self.num_agents*4 : ].reshape(-1, 2)
        
    # Find the region that contains the entity
    def localize(self, entity, language):
        try:
            region_idx = list(map(lambda region: region.contains(entity), language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    def direct(self, approach):
        directions = []
        class_name = approach.__class__.__name__
        
        if class_name in self.langauges:
            obstacles = [Point(obstacle).buffer(self.obstacle_radius) 
                         for obstacle in self.obstacles]
            
            for agent, goal in zip(self.agents, self.goals):
                agent_idx = self.localize(Point(agent), approach)
                goal_idx = self.localize(Point(goal), approach)
                directions += [a_star(agent_idx, goal_idx, obstacles, approach)]
        else:
            for agent, goal in zip(self.agents, self.goals):
                directions += [approach.direct(agent, goal, self.obstacles)]
            
        return directions