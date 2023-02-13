import pdb
import copy
import numpy as np

from math import inf
from agents.utils.search import astar
from agents.utils.networks import SpeakerNet
from agents.utils.languages import LanguageFactory

class Speaker:
    def __init__(self, input_dims, output_dims):
        self.direction_len = 28
        self.speaker = SpeakerNet(input_dims, output_dims)
        
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
        
    # Gather directions from learned language
    def communicate(self, langauge, path, obstacles):
        directions, polygons = language.direct(path, obstacles)
        while len(directions) < self.direction_len:
            directions.insert(0,0)
            
        assert len(directions) == self.direction_len
        return directions, polygons
        
    # Convert directions to regions to ensure listener is in correct region
    def feedback(self, pos, goal, obstacles, polygons, idx):
        a=3        
        
    def learn(self):
        a=3