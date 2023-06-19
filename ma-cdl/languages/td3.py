"""
This code is based on the following repository:

Author: Donal Byrne
Repository: TD3
URL: https://github.com/djbyrne/TD3
Version: 21d162f
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

"""Twin Delayed Deep Deterministic Policy Gradient (TD3)"""
class TD3(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.valid_lines = set()
    
        self.action_dim = 3
        self.action_range = 1
        self.state_dims = 128
        
        self._init_hyperparams()
        self._init_wandb()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer()
        self.autoencoder = AE(self.state_dims, self.rng, self.max_lines)
        
        self.actor = Actor(self.state_dims, self.action_dim, self.action_range)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(self.state_dims, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters())  
        
    def _init_hyperparams(self):
        self.tau = 0.005
        self.gamma = 0.99
        self.batch_size = 64
        self.num_dummy = 150000
        self.noise_clip = 0.025
        self.num_episodes = 1000
        self.policy_noise = 0.005
        self.num_iterations = 100
        self.policy_update_freq = 2
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='TD3')
        config = wandb.config
        config.tau = self.tau
        config.gamma = self.gamma 
        config.weights = self.weights
        config.num_dummy = self.num_dummy
        config.noise_clip = self.noise_clip
        config.batch_size = self.batch_size
        config.policy_noise = self.policy_noise
        config.num_iterations = self.num_iterations
        config.policy_update_freq = self.policy_update_freq
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'episode', episode, regions, reward)
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
    def _remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:            
            default_boundary_lines = CDL.get_valid_lines([])  
            default_square = CDL.create_regions(default_boundary_lines)
            start_state = self.autoencoder.get_state(default_square)
            
            # Hallicinate transitions according to shuffled actions
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
                    reward = self.rewards[idx]
                    done = True if idx == len(shuffled_action) - 1 else False
                    
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
            
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.valid_lines.clear()
            
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        num_action = 1
        state = start_state
        while len(self.replay_buffer) < self.num_dummy:
            action = self.rng.uniform(-self.action_range, self.action_range, size=self.action_dim)
            reward, next_state, done, _ = self._step(problem_instance, action, num_action)
            self._remember(state, action, reward, next_state, done)
            
            if done: 
                state = start_state
                num_action = 1
            else:
                state = next_state
                num_action += 1
                                    
    # Select an action (coefficients of a linear line)
    def _select_action(self, state, noise=0.05):
        state = torch.FloatTensor(state)
        action = self.actor(state).data.numpy().flatten()
        if noise != 0:
            action = (action + self.rng.uniform(0, noise, size=self.action_dim)).clip(-self.action_range, self.action_range)
            
        return action
    
    # Learn from the replay buffer
    def _learn(self):
        for it in range(self.num_iterations):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(-1)
            
            noise = actions.data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.action_range, self.action_range)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.max(target_Q1, target_Q2)
            target_Q = rewards + (dones * self.gamma * target_Q).detach()
            
            current_Q1, current_Q2 = self.critic(states, actions)
            
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if it % self.policy_update_freq == 0:
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
        
        rewards = []     
        for episode in range(self.num_episodes):
            done = False
            num_action = 1
            state = start_state
            while not done: 
                action = self._select_action(state)
                reward, next_state, done, regions = self._step(problem_instance, action, num_action)  
                self._remember(state, action, reward, next_state, done)         
                self._learn()
                state = next_state
                num_action += 1
                
            rewards.append(reward)
            avg_reward = np.mean(rewards[-25:])
                        
            wandb.log({"reward": reward, "avg_reward": avg_reward})
            if episode % 100 == 0 and len(regions) > 0:
                self._log_regions(problem_instance, episode, regions, reward)
    
    def _generate_optimal_coeffs(self, problem_instance):
        self._train(problem_instance)
        
        done = False
        optim_coeffs = []
        with torch.no_grad():
            while not done: 
                action = self._select_action(state, noise=0)
                optim_coeffs.append(action)
                reward, next_state, done, regions = self._step(problem_instance, action)
                state = next_state
        
        wandb.log({"Final Reward": reward})
        self._log_regions(problem_instance, 'final', regions, reward)
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)     
        return optim_coeffs  