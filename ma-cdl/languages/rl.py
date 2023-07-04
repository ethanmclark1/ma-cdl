"""
This code is based on the following repository:

Author: Scott Fujimoto
Repository: TD3
URL: https://github.com/sfujim/TD3/blob/master/TD3.py
Version: 6a9f761
License: MIT License
"""

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
        self.penalty = []
        self.next_states = []
        self.dones = []
        
        self.valid_lines = set()
        
        self.action_dim = 3
        self.state_dim = 128
        
        self._init_hyperparams()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines)
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)  
        
    def _init_hyperparams(self):
        num_records = 8
        
        self.lr = 5e-4
        self.tau = 0.005
        self.gamma = 0.99
        self.policy_freq = 2
        self.batch_size = 128
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.num_episodes = 10000
        self.num_iterations = 100
        self.dummy_transitions = int(5e2)
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='RL')
        config = wandb.config
        config.tau = self.tau
        config.gamma = self.gamma 
        config.weights = self.weights
        config.noise_clip = self.noise_clip
        config.batch_size = self.batch_size
        config.policy_freq = self.policy_freq
        config.policy_noise = self.policy_noise
        config.num_iterations = self.num_iterations
        config.dummy_transitions = self.dummy_transitions
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, penalty):
        pil_image = super()._get_image(problem_instance, 'episode', episode, regions, penalty)
        wandb.log({"image": wandb.Image(pil_image)})
            
    # Overlay lines in the environment
    def _step(self, problem_instance, action, num_action):
        penalty = 0
        done = False
        next_state = []
        prev_num_lines = max(len(self.valid_lines), 4)
                
        line = CDL.get_lines_from_coeffs(action)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_action == self.max_lines:
            done = True
            penalty = -super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
            
        next_state = self.autoencoder.get_state(regions)
        return penalty, next_state, done, regions
    
    # Add transition to replay buffer
    # TODO: Adhere to GitHub TD3 implementation
    def _remember(self, state, action, penalty, next_state, done):
        self.replay_buffer.add(state, action, penalty, next_state, done)
        
        self.states.append(state)
        self.actions.append(action)
        self.penalty.append(penalty)
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
                    penalty = self.penalty[idx]
                    done = True if idx == len(shuffled_action) - 1 else False
                    
                    self.replay_buffer.add(state, action, penalty, next_state, done)
                    state = next_state
            
            self.states = []
            self.actions = []
            self.penalty = []
            self.next_states = []
            self.dones = []
            self.valid_lines.clear()
            
    # Populate replay buffer with dummy transitions
    # TODO: Adhere to GitHub TD3 implementation
    def _populate_buffer(self, problem_instance, start_state):
        num_action = 1
        state = start_state
        for _ in range(self.dummy_transitions):
            action = self.rng.uniform(-1, 1, self.action_dim)
            penalty, next_state, done, _ = self._step(problem_instance, action, num_action)
            self._remember(state, action, next_state, penalty, done)
            
            if done: 
                num_action = 1
                state = start_state
            else:
                num_action += 1
                state = next_state
                                
    # Select an action (coefficients of a linear line)
    def _select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state)
        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + self.rng.uniform(-1, 1, size=self.action_dim)).clip(-1, 1)
            
        return action  
    
    # Learn from the replay buffer
    # TODO: Adhere to TD3 GitHub implementation
    def _learn(self):
        for it in range(self.num_iterations):
            states, actions, penalties, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            with torch.no_grad():
                noise = (2 * torch.rand_like(actions) - 1) * self.policy_noise                
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
                
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = penalties + (dones * self.gamma * target_Q).detach()
            
            current_Q1, current_Q2 = self.critic(states, actions)
            
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
    def _train(self, problem_instance):        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        self._populate_buffer(problem_instance, start_state)
        
        returns = []     
        for episode in range(self.num_episodes):
            done = False
            num_action = 1
            state = start_state
            while not done: 
                action = self._select_action(state)
                penalty, next_state, done, regions = self._step(problem_instance, action, num_action)  
                self._remember(state, action, penalty, next_state, done)         
                self._learn()
                state = next_state
                num_action += 1
                
            returns.append(penalty)
            avg_reward = np.mean(returns[-250:])
            wandb.log({"avg_reward": avg_reward})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, penalty)
    
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        self._train(problem_instance)
        
        done = False
        optim_coeffs = []
        with torch.no_grad():
            num_action = 1
            while not done: 
                action = self._select_action(state, noise=0)
                optim_coeffs.append(action)
                penalty, next_state, done, regions = self._step(problem_instance, action, num_action)
                state = next_state
                num_action += 1
        
        self._log_regions(problem_instance, 'final', regions, penalty)
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)   
        wandb.finish()  
        return optim_coeffs  