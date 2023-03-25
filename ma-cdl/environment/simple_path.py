import random
import numpy as np

from gymnasium.utils import EzPickle
from environment.utils.scenario import BaseScenario
from environment.utils.core import Agent, Landmark, World
from environment.utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, config):
        max_cycles = 200
        continuous_actions = False
        render_mode = 'human'
        scenario = Scenario()
        world = scenario.make_world(config)
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
    def make_world(self, config):
        world = World(config)
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(config.num_obs + 1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = config.obs_size
        return world

    def reset_world(self, world, np_random):
        # obstacles are black
        for i in range(len(world.landmarks)):
            world.landmarks[i].color = np.array([0.75,0.25,0.25])
            
        # random properties for agents
        world.agents[0].goal = world.landmarks[0] # goal is always red landmark
        world.agents[0].color = np.array([0.25, 0.25, 0.75])
        world.agents[0].goal.color = np.array([0.25, 0.75, 0.25])

        # start position is constrained to be in bottom-left quadrant
        world.agents[0].state.p_pos = random.choice(world.start_constr)*np_random.uniform(0, 1, world.dim_p)
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        # goal position is constrained to be in top-left quadrant
        world.landmarks[0].state.p_pos = random.choice(world.goal_constr)*np_random.uniform(0, 1, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        # obstacles are constrained to be in top-right quadrant 
        for landmark in world.landmarks[1:]:
            landmark.state.p_pos = random.choice(world.obs_constr)*np_random.uniform(0, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # Created custom reward function in Speaker class
    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        return np.concatenate((agent.state.p_pos, agent.state.p_vel))