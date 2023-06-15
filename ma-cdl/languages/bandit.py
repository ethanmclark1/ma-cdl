import wandb
import torch
import numpy as np

from torch.optim import Adam
from languages.utils.cdl import CDL
from languages.utils.networks import REINFORCE

"""Infinitely Armed Bandit"""
class Bandit(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.action_dims = 3
        
        self._init_hyperparams()
        self._init_wandb()
        
        self.reinforce = REINFORCE(1, self.action_dims)
        self.optimizer = Adam(self.reinforce.parameters(), lr=self.learning_rate)
    
    def _init_hyperparams(self):
        self.gamma = 0.99
        self.num_episodes = 1000
        self.learning_rate = 0.01
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='bandit')
        config = wandb.config
        config.gamma = self.gamma
        config.num_episodes = self.num_episodes
        config.learning_rate = self.learning_rate
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
    
    # Select action according to context (timestep)
    def _select_action(self, timestep):
        return self.reinforce(timestep)
    
    # Discount rewards for each timestep
    def discount_rewards(self, reward_list):
        discounted_returns = []
        for i in range(len(reward_list)):
            remaining_rewards = reward_list[i:]
            discounts = self.gamma ** np.array(range(len(remaining_rewards)))
            G_t = np.sum(remaining_rewards * discounts)
            discounted_returns.append(G_t)
            
        G = torch.tensor(discounted_returns, dtype=torch.float32)
        return G
    
    def _learn(self, context_list, action_list, reward_list):
        self.optimizer.zero_grad()
        future_return = self.discount_rewards(reward_list)
        G = torch.sum(future_return)
        action_prob = self.reinforce(context_list)
        dist = torch.Multinomial(action_prob)
        log_prob = dist.log_prob(action_list)
        loss = torch.sum(-log_prob * G)
        loss.backward()
        self.optimizer.step()
    
    def _step(self, problem_instance, action, num_action):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)

        line = CDL._get_lines_from_coeffs(action)
        valid_lines = CDL._get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        penalty = -super()._optimizer(regions, problem_instance)
        
        if len(self.valid_lines) == prev_num_lines or num_action == self.max_lines or _reward > self.reward_thres:
            done = True
            reward = _reward
            self.valid_lines.clear()

        return reward, done, regions
        
        
    def _train(self, problem_instance):
        avg_reward = []
        for _ in range(self.num_episodes):
            timesteps = []
            actions = []
            rewards = []
            timestep = 0
            done = False
            while not done:
                action = self._select_action(timestep)
                reward, done, regions = self._step(problem_instance, action, timesteps)
                timesteps.append(timestep)
                actions.append(action)
                rewards.append(reward)
                timestep += 1
            
            self._learn(timesteps, actions, rewards)
        
    def _generate_optimal_coeffs(self, problem_instance):
        self._train(problem_instance)
        
        done = False
        optim_coeffs = []
        with torch.no_grad():
            timestep = 0
            while not done:
                action = self._select_action(timestep)
                optim_coeffs.append(action)
                reward, done = self._step(problem_instance, action)
                timestep += 1
        
        print(f'Final reward: {reward}')
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        return optim_coeffs