import copy
import torch
import numpy as np

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.language import Language

class MA_CDL2():
    def __init__(self, args):
        self.env = simple_path.env(args)
        self.language = Language(self.env)
        self.speaker = Speaker()
        self.listener = Listener()
    
    # Passes language to both speaker and listener
    def create_language(self):
        language = self.language.create()
        self.speaker.set_language(language)
        self.listener.set_language(language)

    def act(self):   
        self.create_language()
        
        start, goal, obstacles = self.env.unwrapped.get_init_conditions()
        directions = self.speaker.direct(start, goal, obstacles)
        obs, _, termination, truncation, _ = self.env.last()
        
        while not (termination or truncation):
            action = self.listener.get_action(obs, goal, directions, self.env)
            self.env.step(action)
            obs, _, termination, truncation, _ = self.env.last()
            
        print('Mission success!' if termination else 'Mission failed!')
        
if __name__ == '__main__':
    args = get_arguments()
    MA_CDL2 = MA_CDL2(args)
    MA_CDL2.act()