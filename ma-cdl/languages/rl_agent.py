from languages.utils.cdl import CDL

class RLAgent(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
    
    def _generate_optimal_coeffs(self, scenario):
        a=3