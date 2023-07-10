"""
This code is based on the following repository:

Author: Scott Fujimoto
Repository: TD3
URL: https://github.com/sfujim/TD3/blob/master/TD3.py
Version: 6a9f761
License: MIT License
"""

import time
import copy
import wandb
import torch
import itertools
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam
from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer

"""Using Twin Delayed Deep Deterministic Policy Gradient (TD3)"""
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.states = []
        self.actions = []
        self.reward = []
        self.next_states = []
        self.dones = []
        
        self.valid_lines = set()
        
        self.action_dim = 3
        self.state_dim = 128
        
        self._init_hyperparams()
        self.rng = np.random.default_rng(seed=42)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines)
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)  
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.01
        self.dummy_eps = 250
        self.policy_freq = 3
        self.discount = 0.999
        self.batch_size = 256
        self.policy_noise = 0.15
        self.num_episodes = 3000
        self.num_iterations = 128
        self.learning_rate = 1e-4
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='RL')
        config = wandb.config
        config.tau = self.tau
        config.discount = self.discount 
        config.weights = self.weights
        config.dummy_eps = self.dummy_eps
        config.batch_size = self.batch_size
        config.policy_freq = self.policy_freq
        config.policy_noise = self.policy_noise
        config.learning_rate = self.learning_rate
        config.num_iterations = self.num_iterations
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
            
    # Overlay lines in the environment
    def _step(self, problem_instance, action, num_action):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
                
        line = CDL.get_lines_from_coeffs(action)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_action == self.max_lines:
            done = True
            reward = -super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
            
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
    
    # Add transition to replay buffer
    def _remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.states.append(state)
        self.actions.append(action)
        self.reward.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:            
            default_boundary_lines = CDL.get_valid_lines([])  
            default_square = CDL.create_regions(default_boundary_lines)
            start_state = self.autoencoder.get_state(default_square)
            
            # Hallucinate transitions according to shuffled actions
            shuffled_actions = list(itertools.permutations(self.actions))
            for shuffled_action in shuffled_actions:
                state = start_state
                self.valid_lines.clear()
                self.valid_lines.update(default_boundary_lines)
                
                for idx, action in enumerate(shuffled_action):
                    line = CDL.get_lines_from_coeffs(action)
                    self.valid_lines.update(CDL.get_valid_lines(line))
                    regions = CDL.create_regions(list(self.valid_lines))
                    next_state = self.autoencoder.get_state(regions)
                    reward = self.reward[idx]
                    done = True if idx == len(shuffled_action) - 1 else False
                    
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
            
            self.states = []
            self.actions = []
            self.reward = []
            self.next_states = []
            self.dones = []
            self.valid_lines.clear()
            
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_eps):
            done = False
            num_action = 1
            state = start_state
            while not done:
                action = self.rng.uniform(-1, 1, self.action_dim)
                reward, next_state, done, _ = self._step(problem_instance, action, num_action)
                self._remember(state, action, reward, next_state, done)
                state = next_state
                num_action += 1
                                
    # Select an action (coefficients of a linear line)
    def _select_action(self, state, add_noise):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        
        if add_noise: 
            noise = self.rng.uniform(-self.policy_noise, self.policy_noise, self.action_dim)
            noise = torch.FloatTensor(noise)
            action = (action + noise).clamp(-1, 1)
                                  
        return action.detach().numpy()
    
    # Learn from the replay buffer
    def _learn(self):
        for it in range(self.num_iterations):
            states, actions, rewards, next_states, not_dones = self.replay_buffer.sample(self.batch_size)
            
            with torch.no_grad():
                noise = self.rng.uniform(-self.policy_noise, self.policy_noise, self.action_dim)
                noise = torch.FloatTensor(noise)
                next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
                
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + not_dones * self.discount * target_Q
            
            current_Q1, current_Q2 = self.critic_target(states, actions)
            
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if it % self.policy_freq == 0:
                actor_loss = -self.critic.get_Q1(states, self.actor(states)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    # Train model on a given problem_instance
    def _train(self, problem_instance, start_state):        
        returns = []     
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            state = start_state
            while not done: 
                num_action += 1
                action = self._select_action(state, add_noise=True)
                reward, next_state, done, regions = self._step(problem_instance, action, num_action)  
                self._remember(state, action, reward, next_state, done)         
                self._learn()
                state = next_state
                
            returns.append(reward)
            avg_returns = np.mean(returns[-100:])
            wandb.log({"Average Returns": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
    
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        start_time = time.time()
        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        
        self._populate_buffer(problem_instance, start_state)
        self._train(problem_instance, start_state)
        
        done = False
        optim_coeffs = []
        with torch.no_grad():
            num_action = 1
            state = start_state
            while not done: 
                action = self._select_action(state, add_noise=False)
                optim_coeffs.append(action)
                reward, next_state, done, regions = self._step(problem_instance, action, num_action)
                state = next_state
                num_action += 1
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        wandb.log({"Elapsed Time": elapsed_time})
        
        self._log_regions(problem_instance, 'Final', regions, reward)
        wandb.finish()  
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)   
        return optim_coeffs  