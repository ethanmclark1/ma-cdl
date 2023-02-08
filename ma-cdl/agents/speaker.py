import pdb
import copy
import numpy as np

from math import inf
from sequitur import quick_train
from sequitur.models import LINEAR_AE
from agents.utils.search import astar
from agents.utils.languages import LanguageFactory

class Speaker:
    def __init__(self, min_symbol, max_symbol):
        self.min_symbol, self.max_symbol = min_symbol, max_symbol
        self.language_set = LanguageFactory(max_symbol)
        
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

    # Gather directions from a random language
    def communicate(self, path, obstacles):
        idx = np.random.randint(self.min_symbol, self.max_symbol+1)
        return self.language_set[2].direct(path, obstacles)

    def feedback(self, pos, goal):    
        a=3