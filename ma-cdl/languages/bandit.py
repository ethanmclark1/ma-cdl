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
        self.action_dim = 3
        self.action_range = 1
        self.valid_lines = set()
        
        self._init_hyperparams()
        self._init_wandb()
        
        self.reinforce = REINFORCE(1, self.action_dim, self.action_range)
        self.optimizer = Adam(self.reinforce.parameters(), lr=self.learning_rate)
    
    def _init_hyperparams(self):
        self.gamma = 0.99
        self.num_episodes = 1000
        self.penalty_thresh = -20
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
    
    # TODO: Figure out how to give immediate reward
    def _step(self, problem_instance, action, num_action):
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)

        line = CDL.get_lines_from_coeffs(action.detach().numpy())
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        penalty = -super().optimizer(regions, problem_instance)
        
        if len(self.valid_lines) == prev_num_lines or num_action == self.max_lines or penalty > self.penalty_thresh:
            done = True
            self.valid_lines.clear()

        return penalty, done, regions
    
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
        mean, std_dev = self.reinforce(context_list)
        normal_distribution = torch.distributions.Normal(mean, std_dev)
        log_prob = normal_distribution.log_prob(action_list)
        loss = torch.sum(-log_prob * G)
        loss.backward()
        self.optimizer.step()

    # Train contextual bandit on a given problem instance
    def _train(self, problem_instance):
        returns = []
        avg_returns = []
        
        for episode in range(self.num_episodes):
            timesteps = []
            actions = []
            rewards = []
            timestep = 0
            done = False
            while not done:
                action = self._select_action(timestep)
                reward, done, regions = self._step(problem_instance, action, timestep)
                timesteps.append(timestep)
                actions.append(action)
                rewards.append(reward)
                timestep += 1
            
            self._learn(timesteps, actions, rewards)
            
            wandb.log({"returns": returns, "avg_returns": avg_returns})
            
            returns.append(np.sum(rewards))
            avg_returns.append(np.mean(returns[-100:]))
            
            if episode % 100 == 0 and len(regions) > 0:
                self._log_regions(problem_instance, episode, regions, reward)
        
    def _generate_optimal_coeffs(self, problem_instance):
        self._train(problem_instance)
        
        done = False
        optim_coeffs = []
        with torch.no_grad():
            timestep = 0
            while not done:
                action = self._select_action(timestep)
                reward, done, regions = self._step(problem_instance, action, timestep)
                optim_coeffs.append(action)
                timestep += 1
        
        wandb.log({"Final Reward": reward})
        self._log_regions(problem_instance, 'final', regions, reward)
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        return optim_coeffs