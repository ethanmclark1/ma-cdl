import pdb
import numpy as np

from gymnasium.utils import EzPickle
from environment.utils.scenario import BaseScenario
from environment.utils.core import Agent, Landmark, World
from environment.utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, max_cycles=25, num_obstacles=3, obstacle_size=0.02, continuous_actions=False, render_mode=None):
        scenario = Scenario()
        world = scenario.make_world(num_obstacles, obstacle_size)
        super().__init__(
            scenario=scenario, 
            world=world, 
            render_mode=render_mode,
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_v2"


env = make_env(raw_env)

class Scenario(BaseScenario):
    def make_world(self, num_obstacles, obstacle_size):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_obstacles + 1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = obstacle_size
        return world

    def reset_world(self, world, np_random):
        # obstacles are black
        for i in range(len(world.landmarks)):
            world.landmarks[i].color = np.array([0.75,0.25,0.25])
            
        # random properties for agents
        world.agents[0].goal = world.landmarks[0] # goal is always red landmark
        world.agents[0].color = np.array([0.25, 0.25, 0.75])
        world.agents[0].goal.color = np.array([0.25, 0.75, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # Created custom reward function in Speaker class
    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        return np.concatenate((agent.state.p_pos, agent.state.p_vel))