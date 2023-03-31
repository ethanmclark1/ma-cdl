import random
import numpy as np

from gymnasium.utils import EzPickle
from environment.utils.scenario import BaseScenario
from environment.utils.core import Agent, Landmark, World
from environment.utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, args):
        max_cycles = 200
        continuous_actions = False
        render_mode = args.render_mode
        scenario = Scenario()
        world = scenario.make_world(args)
        super().__init__(
            scenario=scenario, 
            world=world, 
            render_mode=render_mode,
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "QuadExplore"
        self.metadata["num_obstacles"] = len(self.world.landmarks) - 1

env = make_env(raw_env)

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World(args)
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.1
        # add landmarks (4 obstacles + 1 goal)
        world.landmarks = [Landmark() for i in range(4 + 1)]
        world.landmarks[0].name = "goal"
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].size = 0.1
        for i, landmark in enumerate(world.landmarks[1:]):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = args.obs_size
        return world

    def reset_world(self, world, np_random):
        world.agents[0].goal = world.landmarks[0]
        # agent.color = green; goal.color = blue; obstacles.color = red
        world.agents[0].color = np.array([0.25, 0.75, 0.25])
        world.agents[0].goal.color = np.array([0.25, 0.25, 0.75])
        for landmark in world.landmarks[1:]:
            landmark.color = np.array([0.75, 0.25, 0.25])
            
        # set state of start and goal
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.p_pos = np.random.uniform(*zip(*world.start_constr))
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        world.landmarks[0].state.p_pos = np.random.uniform(*zip(*world.goal_constr))
        
        # set state of obstacles
        if isinstance(world.obs_constr, tuple):
            for i in range(len(world.landmarks[1:])):
                world.landmarks[i+1].state.p_vel = np.zeros(world.dim_p)
                world.landmarks[i+1].state.p_pos = np.random.uniform(*zip(*world.obs_constr))
        else:
            for i in range(len(world.landmarks[1:])):
                if i < len(world.landmarks[1:]) / 2:
                    world.landmarks[i+1].state.p_vel = np.zeros(world.dim_p)
                    world.landmarks[i+1].state.p_pos = np.random.uniform(*zip(*world.obs_constr[0]))
                else:
                    world.landmarks[i+1].state.p_vel = np.zeros(world.dim_p)
                    world.landmarks[i+1].state.p_pos = np.random.uniform(*zip(*world.obs_constr[1]))
            
    # Created custom reward function in Speaker class
    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        return np.concatenate((agent.state.p_pos, agent.state.p_vel))