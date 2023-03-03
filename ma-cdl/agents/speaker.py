import copy
import numpy as np

from math import inf
from shapely import points
from agents.utils.search import astar

class Speaker:
    def __init__(self):
        self.langauge = None
    
    def set_language(self, language):
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
    
    # TODO: Figure out efficient way to get directions as opposed to linearly traversing path
    def direct(self, path, obstacles):
        directions = []
        path = points(path)
        for pos in path:
            region = None
            if region != direction[-1]:
                directions.append(region)
        return directions

