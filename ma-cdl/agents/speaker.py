import copy
import math
import numpy as np

from agents.utils.search import astar
from agents.utils.networks import Planner

class Speaker:
    def __init__(self, language, num_obstacles):
        input_dims = language*4 + num_obstacles*2 + 2 + 2
        self.planner = Planner(input_dims) 
        self.langauge = language
                
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
        directions = self.planner(language, path, obstacles)
        return directions

