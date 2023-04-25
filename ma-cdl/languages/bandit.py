from languages.utils.cdl import CDL

class MultiArmedBandit(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)