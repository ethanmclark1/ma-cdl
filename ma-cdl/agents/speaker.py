import pdb
import copy
import numpy as np

from math import inf
from agents.utils.search import astar
from agents.utils.network import SpeakerNet
from agents.utils.languages import LanguageFactory

class Speaker:
    def __init__(self, input_dims, min_symbol, max_symbol):
        self.direction_len = 28
        self.min_symbol, self.max_symbol = min_symbol, max_symbol
        self.language_set = LanguageFactory(min_symbol, max_symbol)
        self.speaker = SpeakerNet(input_dims, len(self.language_set))
        
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
    
    # Select which representation to use to communicate
    def select(self, start, goal, obstacles, env_shape):
        representation_idx = self.speaker(start, goal, obstacles, env_shape)
        representation_idx = np.argmax(representation_idx.detach().numpy())
        # Add min_symbol to get language_set key value
        representation_idx += self.min_symbol
        return representation_idx
        
    # Gather directions from learned language
    def communicate(self, path, obstacles, idx):
        directions = self.language_set[idx].direct(path, obstacles)
        while len(directions) < self.direction_len:
            directions.insert(0,0)
            
        assert len(directions) == self.direction_len
        return directions

    # Convert directions to regions to ensure listener is in correct region
    def feedback(self, pos, goal, obstacles, directions, idx):
        # Remove padding    
        directions = list(filter(lambda x: x != 0, directions))
        
        
    def learn(self):
        a=3