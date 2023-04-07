import random
import numpy as np

from gymnasium.utils import EzPickle
from environment.utils.scenario import BaseScenario
from environment.utils.core import Agent, Landmark, World
from environment.utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, args):
        max_cycles = 500
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
        self.metadata["agent_radius"] = self.world.landmarks[0].size
        self.metadata["obstacle_radius"] = args.obs_size

env = make_env(raw_env)

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        self.get_problem_configuration(world, args.problem)
        
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
    
    def get_problem_configuration(self, world, problem):
        problem_configurations = {
            'Cluster': {
                'start': ((-1, -0.80), (-0.25, 0.25)),
                'goal': ((0.80, 1), (-0.25, 0.25)),
                'obs': ((-0.15, 0.15), (-0.15, 0.15))
            },
            'L-shaped': {
                'start': ((-1, -0.80), (-0.25, 0.25)),
                'goal': ((0.4, 0.6), (0.4, 0.6)),
                'obs': [((-0.1, 0.1), (0, 0.6)), ((0.1, 0.5), (0, 0.25))]
            },
            'Vertical': {
                'start': ((-1, -0.80), (-1, 1)),
                'goal': ((0.80, 1), (-1, 1)),
                'obs': ((-0.075, 0.075), (-0.6, 0.6))
            },
            'Horizontal': {
                'start': ((-1, 1), (-1, -0.80)),
                'goal': ((-1, 1), (0.80, 1)),
                'obs': ((-0.6, 0.6), (-0.075, 0.75))
            },
            'Left': {
                'start': ((0, 1), (-1, -0.80)),
                'goal': ((0, 1), (0.80, 1)),
                'obs': ((-1, 0), (-1, 1))
            },
            'Right': {
                'start': ((-1, 0), (-1, -0.80)),
                'goal': ((-1, 0), (0.80, 1)),
                'obs': ((0, 1), (-1, 1))
            },
            'Up': {
                'start': ((-1, 0.80), (-1, 0)),
                'goal': ((0.80, 1), (-1, 0)),
                'obs': ((-1, 1), (0, 1))
            },
            'Down': {
                'start': ((-1, 0.80), (0, 1)),
                'goal': ((0.80, 1), (0, 1)),
                'obs': ((-1, 1), (-1, 0))
            },
            'Random': {
                'start': ((-1, 1), (-1, 1)),
                'goal': ((-1, 1), (-1, 1)),
                'obs': ((-1, 1), (-1, 1))
            }
        }
        
        world.possible_problem_types = list(problem_configurations.keys())

        problem_info = problem_configurations[problem.capitalize()]
        world.problem_type = problem
        world.start_constr = problem_info['start']
        world.goal_constr = problem_info['goal']
        world.obs_constr = problem_info['obs']