from shapely import Point
from agents.utils.a_star import a_star
from agents.utils.base_agent import BaseAgent

class Speaker(BaseAgent):
    def __init__(self, num_agents, agent_radius, obstacle_radius, goal_radius):
        super().__init__(agent_radius, goal_radius, obstacle_radius)
        self.agents = []
        self.goals = []
        self.obstacles = None
        self.num_agents = num_agents
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state):
        for idx in range(self.num_agents):
            self.agents += [state[idx*2 : idx*2+2]]
            self.goals += [state[self.num_agents*2 + idx*2 : self.num_agents*2 + idx*2 + 2]]
            
        self.obstacles = state[self.num_agents*4 : ].reshape(-1, 2)
    
    def direct(self, approach):
        directions = []
        class_name = approach.__class__.__name__
        
        if class_name in self.languages:
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