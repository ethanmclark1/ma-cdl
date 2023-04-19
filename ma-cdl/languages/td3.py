import torch
import numpy as np

from torch.optim import Adam
from itertools import chain, permutations
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer
from sklearn.preprocessing import OneHotEncoder
from environment.utils.problems import problem_scenarios

"""Twin Delayed Deep Deterministic Policy Gradients (TD3)"""
class TD3(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles, max_action=2, action_dim=2, steps=6):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self.steps = steps
        scenarios = np.array(list(problem_scenarios.keys())).reshape(-1, 1)
        self.encoded_scenarios = OneHotEncoder().fit_transform(scenarios).toarray()
        
        # linear equation: ax + by = 0
        state_dim = self.encoded_scenarios.shape[1] + len(self.square.exterior.coords) * len(self.square.exterior.coords[0])
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters())        
    
    # Select an action (ceefficients of a linear line)
    def _generate_coeffs(self, state, noise=0.1):
        coeffs = self.actor(state).data.numpy().flatten()
        if noise != 0:
            coeffs = (coeffs + np.random.normal(0, noise, size=self.action_dim))
            
        return coeffs
    
    # Get transitions for replay buffer
    def _get_transition(self, scenario):
        lines, coefficients = set(), []
        scenario_idx = list(problem_scenarios.keys()).index(scenario)
        encoded_scenario = self.encoded_scenarios[scenario_idx].reshape(1, -1)
        regions = np.array(self.square.exterior.coords).reshape(1, -1)

        state = torch.FloatTensor(np.concatenate((encoded_scenario, regions), axis=-1))

        step = 0
        while step < self.steps:
            coeffs = self._generate_coeffs(state)
            line = self._get_lines_from_coeffs(coeffs)
            valid_lines = self._get_valid_lines(line)
            lines.update(valid_lines)
            if len(valid_lines) > 4:
                coefficients.append(coeffs)
            else:
                break
            step += 1
        
        new_regions = self._create_regions(lines)
        new_region_coords = [list(region.exterior.coords) for region in new_regions]
        flattened_regions = np.array(list(chain.from_iterable(new_region_coords))).reshape(1, -1)
        next_state = torch.FloatTensor(np.concatenate((encoded_scenario, flattened_regions), axis=-1))
        reward = -self._optimizer(coefficients, scenario)
        
        return state, coefficients, reward, next_state, True
        
    def _generate_optimal_coeffs(self, scenario):
        replay_buffer = ReplayBuffer(size=100000)
        state, action, reward, next_state, done = self._get_transition(scenario)
        action_ordering = list(permutations(action))
        [replay_buffer.add(state, action_order, reward, next_state, done) for action_order in action_ordering]

        a=3
