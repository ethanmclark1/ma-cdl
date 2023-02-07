import pdb
import copy
import numpy as np

from math import inf
from sequitur import quick_train
from sequitur.models import LINEAR_AE
from agent.utils.search import astar
from agent.utils.languages import LanguageFactory

class Speaker:
    def __init__(self, max_symbols):
        self.language_set = LanguageFactory(max_symbols)
        
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

    # Gather directions from specified language
    def communicate(self, path, obstacles, idx):
        return self.language_set[idx].direct(path, obstacles)

    def feedback(self, pos, goal):    
        a=3