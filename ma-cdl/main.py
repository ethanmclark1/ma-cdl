import copy
import torch
import numpy as np

from arguments import get_arguments
from environment import simple_path
from agents.speaker import Speaker
from agents.listener import Listener
from agents.utils.language import Language

class MA_CDL():
    def __init__(self, args):
        self.env = simple_path.env(
            max_cycles=200, 
            num_obstacles=args.num_obstacles,
            obstacle_size=args.obstacle_size,
            render_mode=args.render_mode
            )
        self.speaker = Speaker()
        self.listener = Listener()
        self.language = Language(args)
    
    # Passes language to both speaker and listener
    def _set_langauge(self, language):
        self.speaker.set_language(language)
        self.listener.set_language(language)

    def act(self):   
        language = self.language.create()
        self._set_langauge(language)
        path, obstacles, backup = self.speaker.search(self.env)
        # Reset environment to initial state
        self.env.unwrapped.steps = 0
        self.env.unwrapped.world = backup
        
        directions = self.speaker.direct(path, obstacles)
        obs, _, termination, truncation, _ = self.env.last()
        while not (termination or truncation):
            action = self.listener.get_action(obs, directions)
            self.env.step(action)
            obs, _, termination, truncation, _ = self.env.last()
        
if __name__ == '__main__':
    args = get_arguments()
    ma_cdl = MA_CDL(args)
    ma_cdl.act()