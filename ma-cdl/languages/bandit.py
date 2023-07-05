"""
This code is based on the following repository:

Author: Scott Fujimoto
Repository: TD3
URL: https://github.com/sfujim/TD3/blob/master/TD3.py
Version: 6a9f761
License: MIT License
"""

import wandb
import torch
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam
from languages.utils.cdl import CDL
from languages.utils.networks import Actor, Critic
from languages.utils.replay_buffer import ReplayBuffer

"""Using Twin Delayed Deep Deterministic Policy Gradient (TD3)"""
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.actions = []
        self.valid_lines = set()

        self.action_dim = 3

        self._init_hyperparams()
        self.rng = np.random.default_rng()
        self.replay_buffer = ReplayBuffer(1, self.action_dim)

        self.actor = Actor(1, self.action_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(1, self.action_dim)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparams(self):
        num_records = 8

        self.lr = 3e-4
        self.dummy_eps = 1
        self.policy_freq = 2
        self.batch_size = 128
        self.policy_noise = 0.2
        self.num_episodes = 10000
        self.num_iterations = 100
        self.record_freq = self.num_episodes // num_records

    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='RL')
        config = wandb.config
        config.weights = self.weights
        config.dummy_eps = self.dummy_eps
        config.noise_clip = self.noise_clip
        config.batch_size = self.batch_size
        config.policy_freq = self.policy_freq
        config.policy_noise = self.policy_noise
        config.num_iterations = self.num_iterations

    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'episode', episode, regions, reward)
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

        return reward, done, regions

    # Add transition to replay buffer
    def _remember(self, action, reward, done):
        self.actions.append(action)

        if done:
            distributed_reward = reward / len(self.actions)
            for action in self.actions:
                done = True if action is self.actions[-1] else False
                self.replay_buffer.add(0, action, distributed_reward, 0, done)

            self.actions = []
            self.valid_lines.clear()

    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance):
        for _ in range(self.dummy_eps):
            done = False
            num_action = 1
            while not done:
                action = self.rng.uniform(-1, 1)
                reward, done, _ = self._step(problem_instance, action, num_action)
                self._remember(action, reward, done)
                num_action += 1

    # Select an action (coefficients of a linear line)
    def _select_action(self, add_noise):
        context = torch.FloatTensor([0])

        if add_noise:
            noise = self.rng.uniform(-self.policy_noise, self.policy_noise)
            noise = torch.FloatTensor(noise)
            action = (self.actor(context) + noise).clamp(-1, 1)

        return action.detach().numpy()

    # Learn from the replay buffer
    def _learn(self):
        for it in range(self.num_iterations):
            states, actions, rewards, _, _ = self.replay_buffer.sample(self.batch_size)
                
            target_Q = rewards
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

    # Train model on a given problem_instance
    def _train(self, problem_instance):
        returns = []
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 1
            while not done:
                action = self._select_action(add_noise=True)
                reward, done, regions = self._step(problem_instance, action, num_action)
                self._remember(action, reward, done)
                self._learn()
                num_action += 1

            returns.append(reward)
            avg_return = np.mean(returns[-250:])
            wandb.log({"avg_return": avg_return})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)

    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()

        self._populate_buffer(problem_instance)
        self._train(problem_instance)

        done = False
        optim_coeffs = []
        with torch.no_grad():
            num_action = 1
            while not done:
                action = self._select_action(add_noise=False)
                optim_coeffs.append(action)
                reward, done, regions = self._step(problem_instance, action, num_action)
                num_action += 1

        self._log_regions(problem_instance, 'Final', regions, reward)
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        wandb.finish()
        return optim_coeffs