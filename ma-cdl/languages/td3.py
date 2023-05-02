"""
This code is based on the following repository:

Author: Donal Byrne
Repository: TD3
URL: https://github.com/djbyrne/TD3
Version: 21d162f
License: MIT License
"""

import os
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
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer

"""Twin Delayed Deep Deterministic Policy Gradient (TD3)"""
class TD3(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self._init_hyperparams()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer(size=self.replay_buffer_size)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_range)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters())
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters())   
        
    def _save_model(self):
        directory = 'ma-cdl/languages/history/saved_models'
        filepaths = [f'{directory}/actor.pth', f'{directory}/critic.pth']
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), filepaths[0])
        torch.save(self.critic.state_dict(), filepaths[1])

    def _load_model(self):
        directory = 'ma-cdl/languages/history/saved_models'
        self.actor.load_state_dict(torch.load(f'{directory}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/critic.pth')) 
    
    # Initialize hyperparameters & set up Weights and Biases
    def _init_hyperparams(self):
        self.lines = []
        self.action_dim = 3
        self.state_dim = 300
        self.max_actions = 6
        self.action_range = 1
        self.states, self.actions, self.rewards, self.next_states, self.dones = \
            [], [], [], [], []
        wandb.init(project='td3', entity='ethanmclark1')
        config = wandb.config
        config.weights = self.weights
        config.tau = self.tau = 0.005
        config.gamma = self.gamma = 0.99
        config.batch_size = self.batch_size = 64
        config.num_dummy = self.num_dummy = 400000
        config.reward_thres = self.reward_thres = -30
        config.policy_noise = self.policy_noise = 0.005
        config.num_episodes = self.num_episodes = 75000
        config.num_iterations = self.num_iterations = 100
        config.policy_update_freq = self.policy_update_freq = 2
        config.replay_buffer_size = self.replay_buffer_size = 800000
    
    # Upload regions to Weights and Biases
    def _log_regions(self, scenario, episode, regions, reward):
        _, ax = plt.subplots()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Scenario: {scenario} \nEpisode: {episode}, \nPenalty: {reward}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)

        wandb.log({"image": wandb.Image(pil_image)})
    
    # Constant pad the state to a fixed size
    def _pad_state(self, state):
        flattened_state = np.array(state).ravel()
        pad_number = self.state_dim - len(flattened_state)
        padded_values = np.zeros(pad_number) - 2
        padded_state = np.hstack((padded_values, flattened_state))
        padded_state = torch.FloatTensor(padded_state)
        return padded_state
        
    # Overlay lines in the environment
    def _step(self, scenario, action, num_action):
        done = False
        reward = -125
        next_state = []
        
        self.lines += self._get_lines_from_coeffs(action)
        regions = self._create_regions(self.lines)
        _reward = -super()._optimizer(regions, scenario)
        
        if num_action == self.max_actions or _reward > self.reward_thres:
            done = True
            reward = _reward
            self.lines = []
        
        next_state.extend([coord for region in regions for coord in region.exterior.coords])
        next_state = self._pad_state(next_state)
        return reward, next_state, done
    
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
                        _reward = -125
                        _done = False               
                             
                    self.replay_buffer.add(_state, _action, _reward, _next_state, _done)
                    _state = _next_state

            self.states, self.actions, self.rewards, self.next_states, self.dones = \
                [], [], [], [], []
    
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, scenario, start_state):
        num_action = 1
        
        state = start_state
        while len(self.replay_buffer) < self.num_dummy:
            action = self.rng.uniform(-self.action_range, self.action_range, size=self.action_dim)
            reward, next_state, done = self._step(scenario, action, num_action)
            self._remember(state, action, reward, next_state, done)
            
            if done: 
                num_action = 0
                state = start_state
            else:
                num_action += 1
                state = next_state
                    
    # Select an action (ceefficients of a linear line)
    def _select_action(self, state, noise=0.005):
        action = self.actor(state).data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.action_dim))
            
        return action
    
    # Learn from the replay buffer
    def _learn(self):
        for it in range(self.num_iterations):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
            
            # Select action according to policy and add clipped noise
            noise = action.data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_range, self.action_range)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * self.gamma * target_Q).detach()
            
            # Current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Critic loss 
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates
            if it % self.policy_update_freq == 0:
                actor_loss = -self.critic.get_Q1(state, self.actor(state)).mean()
                
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
        
        for episode in range(self.num_episodes):
            done = False
            state = start_state
            num_action = 1
            while not done: 
                action = self._select_action(state)
                reward, next_state, done = self._step(scenario, action, num_action)   
                self._remember(state, action, reward, next_state, done)         
                self._learn()
                state = next_state
                num_action += 1
            
            rewards.append(reward)
            avg_reward = np.mean(rewards[-100:])
            print(f'Episode: {episode}\nPenalty: {reward}\nAverage Penalty: {avg_reward}\n', end="")
            
            wandb.log({"reward": reward, "avg_reward": avg_reward})
            if episode % 3000 == 0 and len(regions) > 0:
                self._log_regions(scenario, episode, regions, reward)
    
    def _generate_optimal_coeffs(self, scenario):
        state = self._pad_state(self.square.exterior.coords)
        
        done = False
        optim_coeffs = []
        self._train(scenario, state)
        
        with torch.no_grad():
            while not done: 
                action = self._select_action(state, noise=0)
                optim_coeffs.append(action)
                _, next_state, done = self._step(scenario, action, 1)
                state = next_state
                action_count += 1
        
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)     
        return optim_coeffs  