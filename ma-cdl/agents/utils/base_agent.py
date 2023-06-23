class BaseAgent:
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
        self.languages = ['EA', 'TD3', 'Bandit']
        
    # Find the region that contains the entity
    def localize(self, entity, language):
        try:
            region_idx = list(map(lambda region: region.contains(entity), language)).index(True)
        except:
            region_idx = None
        return region_idx   
        