# This code makes use of the Twin Delayed DDPG (TD3) library:
# Fujimoto, Scott et al. (2021). TD3: Twin Delayed DDPG. GitHub.
# Available at: https://github.com/sfujim/TD3

import copy
import torch
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import PrioritizedReplayBuffer

"""Using Twin-Delayed Deep Deterministic Policy Gradient (TD3) with Prioritized Experience Replay"""
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.action_dim = 3
        self._init_hyperparams()
        
        self.actor = None
        self.actor_target = None
        
        self.critic = None
        self.critic_target = None
        
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines)
                
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 5e-3
        self.gamma = 0.99
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.batch_size = 512
        self.policy_noise = 0.2
        self.actor_alpha = 3e-4
        self.critic_alpha = 6e-4
        self.memory_size = 30000
        self.dummy_episodes = 200
        self.num_episodes = 20000
        self.exploration_noise_start = 0.1
        self.exploration_noise_decay = 0.9999
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.gamma = self.gamma 
        config.batch_size = self.batch_size
        config.noise_clip = self.noise_clip
        config.memory_size = self.memory_size
        config.policy_freq = self.policy_freq
        config.actor_alpha = self.actor_alpha
        config.critic_alpha = self.critic_alpha
        config.policy_noise = self.policy_noise
        config.num_episodes = self.num_episodes
        config.dummy_episodes = self.dummy_episodes
        config.exploration_noise = self.exploration_noise_start
        config.exploration_noise_decay = self.exploration_noise_decay
        
    def _decrement_exploration(self):
        self.exploration_noise *= self.exploration_noise_decay
        self.exploration_noise = min(0.01, self.exploration_noise)
        
    # Select line for a given state based on actor network and add exploration noise
    def _select_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.tensor(state))
            
            noise = self.rng.normal(0, self.exploration_noise, size=self.action_dim)
            action = (action.detach().numpy() + noise).clip(-1, 1)
                
        return action
                                      
    # Learn from the replay buffer following the TD3 algorithm
    def _learn(self):
        actor_loss = None
        self.total_it += 1
        
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        current_Q = torch.min(current_Q1, current_Q2)
        td_error = torch.abs(target_Q - current_Q).detach()
        
        weighted_loss_Q1 = (F.mse_loss(current_Q1, target_Q, reduction='none') * weights).mean()
        weighted_loss_Q2 = (F.mse_loss(current_Q2, target_Q, reduction='none') * weights).mean()
        critic_loss = weighted_loss_Q1 + weighted_loss_Q2
        
        self.critic.optim.zero_grad()
        critic_loss.backward()
        self.critic.optim.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.get_Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor.optim.zero_grad()
            actor_loss.backward()
            self.actor.optim.step()
            
            actor_loss = actor_loss.item()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        td_error = td_error.view(1, -1).squeeze().numpy()
        return actor_loss, critic_loss.item(), td_error, tree_idxs
    
    # Retrieve optimal set lines for a given problem instance from the training
    def _generate_optimal_lines(self, problem_instance):        
        # Start from a blank slate every time
        self.total_it = 0
        self.exploration_noise = self.exploration_noise_start
        self.buffer = PrioritizedReplayBuffer(self.state_dim, self.action_dim, self.memory_size)
        self.actor = Actor(self.state_dim, self.action_dim, self.actor_alpha)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.state_dim, self.action_dim, self.critic_alpha)
        self.critic_target = copy.deepcopy(self.critic)
        
        optim_lines = super()._generate_optimal_lines(problem_instance)
        
        return optim_lines