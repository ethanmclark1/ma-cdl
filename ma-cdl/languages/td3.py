"""
This code is based on the following repository:

Author: Donal Byrne
Repository: TD3
URL: https://github.com/djbyrne/TD3
Version: 21d162f
License: MIT License
"""

import io        
import copy
import wandb
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch.optim import Adam
from languages.utils.cdl import CDL
from languages.utils.replay_buffer import ReplayBuffer
from languages.utils.networks import Autoencoder, Actor, Critic

"""Twin Delayed Deep Deterministic Policy Gradient (TD3)"""
class TD3(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.lines = []
        
        self._init_hyperparams()
        self._init_wandb()
        self.rng = np.random.default_rng(seed=42)
        self.replay_buffer = ReplayBuffer(size=self.replay_buffer_size)
        
        input_dims = (self.input_dims[0] * self.input_dims[1])
        self.autoencoder = Autoencoder(input_dim=input_dims, output_dim=self.state_dim)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_range)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters())  
        
    def _init_hyperparams(self):
        self.tau = 0.005
        self.gamma = 0.99
        self.action_dim = 3
        self.batch_size = 64
        self.state_dim = 300
        self.max_actions = 6
        self.action_range = 1
        self.num_dummy = 375000
        self.noise_clip = 0.025
        self.reward_thres = -20
        self.num_episodes = 1000
        self.policy_noise = 0.005
        self.num_iterations = 100
        self.input_dims = (84, 84)
        self.policy_update_freq = 2
        self.replay_buffer_size = 1000000
        
    def _init_wandb(self):
        wandb.init(project='td3', entity='ethanmclark1')
        config = wandb.config
        config.tau = self.tau
        config.gamma = self.gamma 
        config.weights = self.weights
        config.num_dummy = self.num_dummy
        config.noise_clip = self.noise_clip
        config.batch_size = self.batch_size
        config.reward_thres = self.reward_thres
        config.policy_noise = self.policy_noise
        config.num_iterations = self.num_iterations
        config.policy_update_freq = self.policy_update_freq
        config.replay_buffer_size = self.replay_buffer_size
    
    # Upload regions to Weights and Biases
    def _log_regions(self, scenario, episode, regions, reward):
        _, ax = plt.subplots()
        scenario = scenario.capitalize()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Scenario: {scenario}   Episode: {episode}   Reward: {reward:.2f}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)

        wandb.log({"image": wandb.Image(pil_image)})
        
    def _display(self, regions):
        _, ax = plt.subplots()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        plt.show()
    
    # Pixelate the environment
    def _get_state(self, regions):
        _, ax = plt.subplots()
        for region in regions:
            ax.plot(*region.exterior.xy)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf).convert('L')
        pixel_array = np.array(im)
        pixel_array = np.array(Image.fromarray(pixel_array).resize((84, 84)))
        plt.clf()
        plt.close()
        
        state = self.autoencoder(pixel_array).detach()
        return state
            
    # Overlay lines in the environment
    def _step(self, scenario, action):
        done = False
        reward = -1e4
        next_state = []
        
        lines = self._get_lines_from_coeffs(action)
        self.lines.extend(lines)
        regions = self._create_regions(self.lines)
        _reward = -super()._optimizer(regions, scenario)
        
        if len(lines) == 0 or _reward > self.reward_thres:
            done = True
            reward = _reward
            self.lines = []
        
        next_state.extend(coord for region in regions for coord in region.exterior.coords)
        next_state = self._pad_state(next_state)
        return reward, next_state, done, regions
    
    # Shuffle actions to remove order dependency
    def _remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:
            shuffled_actions = list(itertools.permutations(self.actions))
            for shuffled_action in shuffled_actions:
                _state = self.states[0]
                for action_idx, _action in enumerate(shuffled_action):
                    _next_state = []
                    self.lines += self._get_lines_from_coeffs(_action)
                    regions = self._create_regions(self.lines)
                    _next_state.extend([coord for region in regions for coord in region.exterior.coords])
                    _next_state = self._pad_state(_next_state)
                    
                    if action_idx == len(shuffled_action) - 1:
                        _reward = self.rewards[-1]
                        self.lines = []
                        _done = True
                    else:
                        _reward = -1e4
                        _done = False               
                             
                    self.replay_buffer.add(_state, _action, _reward, _next_state, _done)
                    _state = _next_state

            self.states, self.actions, self.rewards, self.next_states, self.dones = \
                [], [], [], [], []
    
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, scenario, start_state):
        state = start_state
        while len(self.replay_buffer) < self.num_dummy:
            action = self.rng.uniform(-self.action_range, self.action_range, size=self.action_dim)
            reward, next_state, done, _ = self._step(scenario, action)
            self._remember(state, action, reward, next_state, done)
            
            state = start_state if done else next_state
                    
    # Select an action (coefficients of a linear line)
    def _select_action(self, state, noise=0.05):
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
            
            # Select actions according to policy and add clipped noise
            noise = actions.data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.action_range, self.action_range)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.max(target_Q1, target_Q2)
            target_Q = rewards + (dones * self.gamma * target_Q).detach()
            
            # Current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)
            
            # Critic loss 
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
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
    
    # Train the model on a given scenario
    def _train(self, scenario, start_state):   
        rewards = []     
        self._populate_buffer(scenario, start_state)
        
        for episode in range(1000):
            done = False
            state = start_state
            while not done: 
                action = self._select_action(state)
                reward, next_state, done, regions = self._step(scenario, action)  
                self._remember(state, action, reward, next_state, done)         
                self._learn()
                state = next_state
                
            rewards.append(reward)
            avg_reward = np.mean(rewards[-25:])
            
            print(f'Episode: {episode}\t Reward: {reward:.2f}\t Average Reward: {avg_reward:.2f}')
            
            wandb.log({"reward": reward, "avg_reward": avg_reward})
            if episode % 100 == 0 and len(regions) > 0:
                self._log_regions(scenario, episode, regions, reward)
    
    def _generate_optimal_coeffs(self, scenario):
        state = self._get_state([self.square])
        
        done = False
        optim_coeffs = []
        self._train(scenario, state)
        
        with torch.no_grad():
            while not done: 
                action = self._select_action(state, noise=0)
                optim_coeffs.append(action)
                reward, next_state, done, _ = self._step(scenario, action)
                state = next_state
        
        print(f'Final reward: {reward}')
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)     
        return optim_coeffs  