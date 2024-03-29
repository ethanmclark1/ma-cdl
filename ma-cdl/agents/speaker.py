import networkx as nx

from shapely import Point
from languages.utils.cdl import CDL

class Speaker:
    def __init__(self, num_agents, obstacle_radius):
        self.num_agents = num_agents
        self.obstacle_radius = obstacle_radius
    
    # Determine the positions of the agents, goals, and obstacles
    def gather_info(self, state):
        self.agents = []
        self.goals = []
        self.obstacles = None
        
        for idx in range(self.num_agents):
            self.agents += [state[idx*2 : idx*2+2]]
            self.goals += [state[self.num_agents*2 + idx*2 : self.num_agents*2 + idx*2 + 2]]
            
        self.obstacles = state[self.num_agents*4 : ].reshape(-1, 2)
    
    # Get the directions for each agent
    def direct(self, approach):
        directions = []
        if isinstance(approach, list):
            obstacles = [Point(obstacle).buffer(self.obstacle_radius) for obstacle in self.obstacles]

            for agent, goal in zip(self.agents, self.goals):
                agent_idx = CDL.localize(agent, approach)
                goal_idx = CDL.localize(goal, approach)
                
                safe_graph = CDL.get_safe_graph(approach, obstacles)
                
                try:
                    directions += [nx.astar_path(safe_graph, agent_idx, goal_idx)]
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    directions += [None]
            
        else:
            for agent, goal in zip(self.agents, self.goals):
                directions += [approach.direct(agent, goal, self.obstacles)]
        
        return directions