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
            backup = copy.deepcopy(self.env.unwrapped.world)
            action = self.listener.get_action(obs, goal, directions, self.env)
            self.env.step(action)
            obs, _, termination, truncation, _ = self.env.last()
        
if __name__ == '__main__':
    args = get_arguments()
    ma_cdl = MA_CDL(args)
    ma_cdl.act()