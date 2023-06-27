import numpy as np

class Agent:
    def __init__(self, radius):
        self.radius = radius
        

class Goal:
    def __init__(self, radius):
        self.radius = radius
        

class Obstacle:
    def __init__(self, obs, size):
        self.x, self.y = obs
        self.size = size
        

class PotentialField:
    def __init__(self):
        self.gain_attr = 0.2
        self.gain_rep = 0.2

    def calc_input(self, cur_pos, target_pos, obstacles):
        term_attr = self._calc_attractive_term(cur_pos, target_pos)
        term_rep = self._calc_repulsive_term(obstacles)
        input = self.gain_attr * term_attr + self.gain_rep * term_rep
        return input

    # obstacle positions are relative distance to the agent
    def _calc_repulsive_term(self, obstacles):
        term = 0
        if len(obstacles) > 0:
            obs_arr = np.array([(obs.x, obs.y, obs.size) for obs in obstacles])
            dist_sqr_arr = obs_arr[:,0] ** 2 + obs_arr[:,1] ** 2 / obs_arr[:,2] ** 2
            denom_arr = dist_sqr_arr ** 1.5
            term = np.sum(np.vstack([obs_arr[:,0], obs_arr[:,1]]) / denom_arr, axis=1)
        return term

    def _calc_attractive_term(self, cur_pos, target_pos):
        term = np.zeros(2)
        cur_x, cur_y = cur_pos
        target_x, target_y = target_pos
        term[0] = target_x - cur_x
        term[1] = target_y - cur_y
        return term


class PathPlanner:
    def __init__(self, agent_radius, goal_radius, obs_radius, max_observable_dist):
        self.dt = 0.1
        self.agent = Agent(agent_radius)
        self.goal = Goal(goal_radius)
        self.obs_radius = obs_radius
        self.potential_field = PotentialField()
        self.max_observable_dist = max_observable_dist
        
    def get_action(self, cur_pos, target_pos, obstacles):        
        obstacles = [Obstacle(obs, self.obs_radius) for obs in obstacles 
                     if (obs < self.max_observable_dist).all()]
        
        new_state = self.potential_field.calc_input(cur_pos, target_pos, obstacles)
        direction = new_state / np.linalg.norm(new_state)

        if np.abs(direction[0]) > np.abs(direction[1]):
            if direction[0] > 0:
                action = 2 # move right
            else:
                action = 1 # move left
        else:
            if direction[1] > 0:
                action = 4 # move up
            else:
                action = 3 # move down

        return action