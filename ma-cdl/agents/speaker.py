import pdb
import copy
import numpy as np

from math import inf
from itertools import product
from scipy.optimize import minimize
from agents.utils.search import astar
from agents.utils.networks import Planner
from agents.utils.language import Language

class Speaker:
    def __init__(self, num_languages, num_obstacles):
        self.num_languages = num_languages
        self.num_obstacles = num_obstacles
    
    """
    Optimization criteria:
        1. Probability of avoiding obstacles
            - Probability of entering obstacle region * probability of colliding with obstacle in obstacle region
        2. Variance on region size
            - Minimize error on all region's size
        3. Amount of navigable space
            - Summed area of regions in which there are no obstacles
        4. Variance on navigable space across problems
            - Compare across multiple problems (i.e. multiple obstacles, different sized obstacles)
    """
    def _optimizer(self, lines):
        # Obstacle(s) constrained to be in top right quadrant
        obs_pos = np.random.rand(self.num_obstacles, 2)
        obs_list = [obs_pos[i] for i in range(self.num_obstacles)]
        return a
    
    def _generate_lines(self):
        bounds = (-1, 1)
        optim_val, optim_coeffs = inf, None
        for num in range(2, self.num_languages+2):
            x0 = (bounds[1] - bounds[0])*np.random.rand(num, 3)+bounds[0]
            res = minimize(self._optimizer, x0, method='nelder-mead',
                           options={'xatol': 1e-8})
            
            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs
    
    def create_language(self):
        lines = self._generate_lines()
        language = Language(lines)
        return language
        
    # Find optimal path using A* search
    def search(self, env):
        path = None
        while not path:
            env.reset()
            max_cycles = env.unwrapped.max_cycles
            world = env.unwrapped.world
            backup = copy.deepcopy(world)
            listener = world.agents[0]
            goal = listener.goal
            obstacles = copy.copy(world.landmarks)
            obstacles.remove(goal)
            env.unwrapped.max_cycles = inf
            path = astar(listener, goal, obstacles, env)
            
        env.unwrapped.max_cycles = max_cycles
        obstacles = np.array([obstacle.state.p_pos for obstacle in obstacles])
        return np.array(path), obstacles, backup
    
    def direct(self, path, obstacles):
        a=3

