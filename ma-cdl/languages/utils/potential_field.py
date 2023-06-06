import numpy as np

class Agent:
    def __init__(self, radius):
        self.radius = radius
        
    def set_state(self, start_pos):
        self.x, self.y = start_pos
        self.u_x = 0.0
        self.u_y = 0.0
        
        self.traj_x = [self.x]
        self.traj_y = [self.y]

    def update_state(self, u_x, u_y, dt):
        self.u_x = u_x
        self.u_y = u_y
        self.x += u_x * dt
        self.y += u_y * dt
        self.traj_x.append(self.x)
        self.traj_y.append(self.y)

        return self.x, self.y

class Goal:
    def __init__(self, radius):
        self.radius = radius
        self.traj_g_x = []
        self.traj_g_y = []
        
    def set_state(self, goal_pos):
        self.x, self.y = goal_pos
        self.traj_g_x.append(self.x)
        self.traj_g_y.append(self.y)

    def get_pos(self):
        return self.x, self.y

class Obstacle:
    def __init__(self, obs, size=None):
        # Static obstacle
        if size is not None:
            self.x, self.y = obs
            self.size = size
        # Dynamic obstacle
        else:
            self._obs = obs
                
    @property
    def x(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                return self._obs.state.p_pos[0]
        else:
            return self._x
        
    @property
    def y(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                return self._obs.state.p_pos[1]
        else:
            return self._y
    
    @property
    def size(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                return self._obs.size
        else:
            return self._size
        
    @x.setter
    def x(self, value):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._obs.state.p_pos[0] = value
        else:
            self._x = value
    
    @y.setter
    def y(self, value):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._obs.state.p_pos[1] = value
        else:
            self._y = value
    
    @size.setter
    def size(self, value):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._obs.size = value
        else:
            self._size = value

class PotentialField:
    def __init__(self):
        self.gain_attr = 0.2
        self.gain_rep = 0.2

    def calc_input(self, goal_x, goal_y, state, obstacles):
        term_attr = self._calc_attractive_term(goal_x, goal_y, state)
        term_rep = self._calc_repulsive_term(state, obstacles)
        input = self.gain_attr * term_attr + self.gain_rep * term_rep
        return input[0], input[1]

    def _calc_repulsive_term(self, state, obstacles):
        term = np.zeros(2)
        for obs in obstacles:
            dist_sqrd = (state.x - obs.x) ** 2 + (state.y - obs.y) ** 2 / obs.size ** 2
            denom = dist_sqrd ** 1.5
            term[0] += (state.x - obs.x) / denom
            term[1] += (state.y - obs.y) / denom
        return term

    def _calc_attractive_term(self, goal_x, goal_y, state):
        term = np.zeros(2)
        term[0] = goal_x - state.x
        term[1] = goal_y - state.y
        return term
    

class PathPlanner:
    def __init__(self, agent_radius, goal_radius, obs_radius):
        self.dt = 0.1
                        
        self.agent = Agent(agent_radius)
        self.goal = Goal(goal_radius)
        self.obs_radius = obs_radius
        self.potential_field = PotentialField()
        
    def get_path(self, start, goal, static_obs, dynamic_obs):
        path = None
        self.agent.set_state(start)
        self.goal.set_state(goal)
        self.obstacles = [Obstacle(obs) for obs in dynamic_obs]
        self.obstacles += [Obstacle(obs, self.obs_radius) for obs in static_obs]
    
        goal_x, goal_y = self.goal.get_pos()
        goal_thresh = self.agent.radius + self.goal.radius
        max_timestep = 500

        time_step = 0
        while True:
            u_x, u_y = self.potential_field.calc_input(goal_x, goal_y, self.agent, self.obstacles)

            self.agent.update_state(u_x, u_y, self.dt)

            dist_to_goal = np.sqrt((goal_x - self.agent.x) ** 2 + (goal_y - self.agent.y) ** 2)

            if dist_to_goal < goal_thresh:
                path = np.vstack((self.agent.traj_x, self.agent.traj_y)).T
                break
                
            time_step += 1
            if time_step >= max_timestep:
                break

        return path