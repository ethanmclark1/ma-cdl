import numpy as np
import matplotlib.pyplot as plt

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
        self.traj_x = []
        self.traj_y = []
        # Static obstacle
        if size is not None:
            self._x, self._y = obs
            self._size = size
            self.traj_x.append(self._x)
            self.traj_y.append(self._y)
        # Dynamic obstacle
        else:
            self._obs = obs
            self._x, self._y = self._obs.state.p_pos
            self._size = self._obs.size

    @property
    def x(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._x = self._obs.state.p_pos[0]
        return self._x

    @property
    def y(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._y = self._obs.state.p_pos[1]
        return self._y

    @property
    def size(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self._size = self._obs.size
        return self._size

    @x.setter
    def x(self, value):
        self._x = value
        self.traj_x.append(value)

    @y.setter
    def y(self, value):
        self._y = value
        self.traj_y.append(value)

    @size.setter
    def size(self, value):
        self._size = value

    def update(self):
        if hasattr(self, '_obs'):
            with self._obs.lock:
                self.x = self._obs.state.p_pos[0]
                self.y = self._obs.state.p_pos[1]
                self.size = self._obs.size


class PotentialField:
    def __init__(self):
        self.gain_attr = 0.25
        self.gain_rep = 0.25

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
        self.dt = 0.15
                        
        self.agent = Agent(agent_radius)
        self.goal = Goal(goal_radius)
        self.obs_radius = obs_radius
        self.potential_field = PotentialField()
    
    # TODO: Make sure dynamic obstacles are updated
    def get_path(self, start, goal, static_obs, dynamic_obs, visualize=True):
        path = None
        self.agent.set_state(start)
        self.goal.set_state(goal)
        self.obstacles = [Obstacle(obs) for obs in dynamic_obs]
        self.obstacles += [Obstacle(obs, self.obs_radius) for obs in static_obs]
    
        goal_x, goal_y = self.goal.get_pos()
        goal_thresh = self.agent.radius + self.goal.radius
        max_timestep = 250

        time_step = 0
        while True:
            [obs.update() for obs in self.obstacles]
            u_x, u_y = self.potential_field.calc_input(goal_x, goal_y, self.agent, self.obstacles)

            self.agent.update_state(u_x, u_y, self.dt)

            dist_to_goal = np.sqrt((goal_x - self.agent.x) ** 2 + (goal_y - self.agent.y) ** 2)

            if dist_to_goal < goal_thresh:
                path = np.vstack((self.agent.traj_x, self.agent.traj_y)).T
                break
                
            time_step += 1
            if time_step >= max_timestep:
                break

        if visualize and path is not None:
            self._visualize(path)
        return path
    
    def _visualize(self, path):
        plt.plot(*path.T, color='cyan', linewidth=2)

        plt.scatter(self.agent.traj_x[0], self.agent.traj_y[0], color='blue', s=100)
        plt.plot(self.agent.traj_x, self.agent.traj_y, color='yellow', linewidth=2)
        plt.scatter(self.goal.traj_g_x, self.goal.traj_g_y, color='green', s=100)

        for obs in self.obstacles:
            plt.scatter(obs.traj_x, obs.traj_y, color='red', s=50)
            if hasattr(obs, '_obs'):
                plt.plot(obs.traj_x, obs.traj_y, color='red', linewidth=2)

        plt.xlim(-1, 1) 
        plt.ylim(-1, 1)
        plt.show()
