# This code makes use of the Twin Delayed DDPG (TD3) library:
# Fujimoto, Scott et al. (2021). TD3: Twin Delayed DDPG. GitHub.
# Available at: https://github.com/sfujim/TD3

import time
import copy
import wandb
import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam
from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer

"""Using Twin-Delayed Deep Deterministic Policy Gradient (TD3)"""
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.states = []
        self.coeffs = []
        self.next_states = []
        
        self.valid_lines = set()
        
        self.action_dim = 3
        self.state_dim = 128
        
        self._init_hyperparams()
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines)
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.alpha)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.alpha)  
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 5e-3
        self.alpha = 3e-5
        self.gamma = 0.99
        self.dummy_eps = 50
        self.batch_size = 64
        self.policy_freq = 2
        self.noise_clip = 0.05
        self.policy_noise = 0.02
        self.num_episodes = 2000
        self.exploration_noise = 0.02
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'RL/{problem_instance.capitalize()}')
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.dummy_eps = self.dummy_eps
        config.batch_size = self.batch_size
        config.noise_clip = self.noise_clip
        config.policy_freq = self.policy_freq
        config.policy_noise = self.policy_noise
        config.num_episodes = self.num_episodes
        config.exploration_noise = self.exploration_noise
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
            
    # Overlay lines in the environment
    def _step(self, problem_instance, mapped_coeffs, num_lines):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
                
        line = CDL.get_lines_from_coeffs(mapped_coeffs)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_lines == self.max_lines:
            done = True
            reward = super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
            
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
    
    # Hallucinate transitions according to shuffled coeffs
    def _hallucinate(self):
        if len(self.coeffs) > 4:
            shuffled_actions = []
            for _ in range(24):
                coeffs = self.coeffs.copy()
                random.shuffle(coeffs)
                shuffled_actions.append(coeffs)
        else:
            shuffled_actions = list(itertools.permutations(self.coeffs))

        return shuffled_actions
    
    # Add transition to replay buffer
    def _remember(self, state, coeffs, reward, next_state, done):        
        self.states.append(state)
        self.coeffs.append(coeffs)
        self.next_states.append(next_state)
        
        if done:            
            default_boundary_lines = CDL.get_valid_lines([])  
            default_square = CDL.create_regions(default_boundary_lines)
            start_state = self.autoencoder.get_state(default_square)
            
            shuffled_coefficients = self._hallucinate()
            for shuffled_coeffs in shuffled_coefficients:
                state = start_state
                self.valid_lines.clear()
                self.valid_lines.update(default_boundary_lines)
                
                for idx, coeffs in enumerate(shuffled_coeffs):
                    mapped_coeffs = CDL.get_mapped_coeffs(coeffs)
                    line = CDL.get_lines_from_coeffs(mapped_coeffs)
                    self.valid_lines.update(CDL.get_valid_lines(line))
                    regions = CDL.create_regions(list(self.valid_lines))
                    next_state = self.autoencoder.get_state(regions)
                    _reward = reward if idx == len(shuffled_coeffs) - 1 else 0
                    done = True if idx == len(shuffled_coeffs) - 1 else False
                    
                    self.replay_buffer.add(state, coeffs, _reward, next_state, done)
                    state = next_state
            
            self.states = []
            self.coeffs = []
            self.next_states = []
            self.valid_lines.clear()
            
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_eps):
            done = False
            num_lines = 0
            state = start_state
            while not done:
                num_lines += 1
                coeffs = self.rng.uniform(-0.1, 0.1, self.action_dim)
                mapped_coeffs = CDL.get_mapped_coeffs(coeffs)
                reward, next_state, done, _ = self._step(problem_instance, mapped_coeffs, num_lines)
                self._remember(state, coeffs, reward, next_state, done)
                state = next_state
                                
    # Select coefficients for a given state
    def _select_coeffs(self, state, add_noise):
        state = torch.FloatTensor(state)
        coeffs = self.actor(state)
        
        if add_noise: 
            noise = self.rng.normal(0, 0.1 * self.exploration_noise, size=self.action_dim)
            coeffs = (coeffs.detach().numpy() + noise).clip(-0.1, 0.1)
            
        mapped_coeffs = CDL.get_mapped_coeffs(coeffs)
        return coeffs, mapped_coeffs
                                      
    # Learn from the replay buffer
    def _learn(self):
        self.total_it += 1
        
        state, action, reward, next_state, not_done = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-0.1, 0.1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
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
    
    # Train model on a given problem_instance
    def _train(self, problem_instance, start_state):        
        returns = []     
        self.total_it = 0
        best_coeffs = None
        best_regions = None
        best_reward = -np.inf

        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            state = start_state
            mapped_coeffs_lst = []
            while not done: 
                num_lines += 1
                coeffs, mapped_coeffs = self._select_coeffs(state, add_noise=True)
                reward, next_state, done, regions = self._step(problem_instance, mapped_coeffs, num_lines)  
                self._remember(state, coeffs, reward, next_state, done)         
                self._learn()
                state = next_state
                mapped_coeffs_lst.append(mapped_coeffs)
                        
            returns.append(reward)
            avg_returns = np.mean(returns[-50:])
            wandb.log({"Average Returns": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
                
            if reward > best_reward:
                best_coeffs = mapped_coeffs_lst
                best_regions = regions
                best_reward = reward
        
        return best_coeffs, best_regions, best_reward
    
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb(problem_instance)
        
        start_time = time.time()
        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        self._populate_buffer(problem_instance, start_state)
        best_coeffs, best_regions, best_reward = self._train(problem_instance, start_state)
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        
        self._log_regions(problem_instance, 'Final', best_regions, best_reward)
        
        wandb.log({"Final Reward": best_reward})
        wandb.log({"Best Coeffs": best_coeffs})
        wandb.log({"Best Num Lines": len(best_coeffs)})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()  
        
        optim_coeffs = np.array(best_coeffs).reshape(-1, self.action_dim)   
        return optim_coeffs  