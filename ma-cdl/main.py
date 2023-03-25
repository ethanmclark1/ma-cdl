import copy
import torch
import numpy as np

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language

class MA_CDL():
    def __init__(self, config):
        self.env = simple_path.env(config)
        self.language = Language(config)
        self.speaker = Speaker()
        self.listener = Listener()
    
    # Passes language to both speaker and listener
    def _set_langauge(self):
        language = self.language.create()
        self.speaker.set_language(language)
        self.listener.set_language(language)
        
    def _get_init_conditions(self):
        world = self.env.unwrapped.world
        start_pos = world.agents[0].state.p_pos
        goal_pos = world.agents[0].goal.state.p_pos
        obstacles = copy.copy(world.landmarks)
        obstacles.remove(world.agents[0].goal)
        obstacles = np.array([obstacle.state.p_pos for obstacle in obstacles])
        return start_pos, goal_pos, obstacles

    def act(self):   
        self.env.reset()
        self._set_langauge()
        
        start, goal, obstacles = self._get_init_conditions()
        directions = self.speaker.direct(start, goal, obstacles)
        obs, _, termination, truncation, _ = self.env.last()
        
        while not (termination or truncation):
            action = self.listener.get_action(obs, goal, directions, self.env)
            self.env.step(action)
            obs, _, termination, truncation, _ = self.env.last()
            
        print('Mission success!' if termination else 'Mission failed!')
        
if __name__ == '__main__':
    config = get_arguments()
    ma_cdl = MA_CDL(config)
    ma_cdl.act()