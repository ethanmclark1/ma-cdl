"""
This code is based on the following repository:

Author: Donal Byrne
Repository: TD3
URL: https://github.com/djbyrne/TD3
Version: 21d162f
License: MIT License
"""

import copy
import torch
import numpy as np

from torch.optim import Adam
from torch.nn.functional import mse_loss
from itertools import chain, permutations
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer
from sklearn.preprocessing import OneHotEncoder
from environment.utils.problems import problem_scenarios

""" Twin Delayed Deep Deterministic Policy Gradient (TD3) """
class TD3(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self._init_hyperparameters()
        self.encoder = OneHotEncoder()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer(size=self.replay_buffer_size)
        scenarios = np.array(list(problem_scenarios.keys())).reshape(-1, 1)
        self.encoded_scenarios = self.encoder.fit_transform(scenarios).toarray()
        
        self.max_action = 2
        self.action_dim = 14
        state_dim = self.encoded_scenarios.shape[1] + len(self.square.exterior.coords) * len(self.square.exterior.coords[0])
        self.actor = Actor(state_dim, self.action_dim, self.max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters())    
    
    # Initialize hyperparameters
    def _init_hyperparameters(self):
        self.tau = 0.005
        self.gamma = 0.99
        self.batch_size = 100
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.num_episodes = 750
        self.policy_noise = 0.2
        self.replay_buffer_size = 25000
    
    # Fill the replay buffer with dummy transitions
    def _populate_buffer(self):
        start_regions = np.array(self.square.exterior.coords).reshape(1, -1)
        while len(self.replay_buffer) < self.replay_buffer_size:
            new_lines, actions = set(), []
            encoded_scenario = self.rng.choice(self.encoded_scenarios, size=1).flatten()
            scenario = self.encoder.inverse_transform(encoded_scenario.reshape(1, -1)).item()
        
            coeffs = self.rng.uniform(-1, 1, size=self.action_dim)
            generated_lines = self._get_lines_from_coeffs(coeffs)
            for idx, line in enumerate(generated_lines):
                valid_lines = self._get_valid_lines([line])
                
                new_lines.update(valid_lines)
                if len(valid_lines) > 4:
                    actions.append(coeffs[idx:idx+3])
                else:
                    break
            
            state = torch.FloatTensor(np.concatenate((encoded_scenario, start_regions), axis=-1))
            new_regions = self._create_regions(new_lines)
            new_regions_coords = [list(region.exterior.coords) for region in new_regions]
            flattened_regions = np.array(list(chain.from_iterable(new_regions_coords))).reshape(1, -1).flatten()
            next_state = torch.FloatTensor(np.concatenate((encoded_scenario, flattened_regions), axis=-1))
            reward = -self._optimizer(actions, scenario)
            
            action_ordering = list(permutations(actions))
            [self.replay_buffer.add(state, action_order, reward, next_state) for action_order in action_ordering]
            
    def _train(self):
        for episode in range(self.num_episodes):
            state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward)
            next_state = torch.FloatTensor(next_state)
            
            # Next state is terminal, therefore target_Q is the reward
            target_Q = reward
            
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = mse_loss(current_Q1, target_Q) + mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if episode % self.policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            
        
            
    
    # Select an action (ceefficients of a linear line)
    def _select_coeffs(self, state, noise=0.1):
        coeffs = self.actor(state).data.numpy().flatten()
        if noise != 0:
            coeffs = (coeffs + np.random.normal(0, noise, size=self.action_dim))
            
        return coeffs 
    
    # Calls the train function to train a model, then returns the coefficients from inference on the model
    def _generate_optimal_coeffs(self, scenario):
        scenario_idx = list(problem_scenarios.keys()).index(scenario)
        encoded_scenario = self.encoded_scenarios[scenario_idx].reshape(1, -1)
        regions = np.array(self.square.exterior.coords).reshape(1, -1)
        state = torch.FloatTensor(np.concatenate((encoded_scenario, regions), axis=-1))
        
        self._populate_buffer()
        self._train()
        coeffs = self._select_coeffs(state)
        return coeffs