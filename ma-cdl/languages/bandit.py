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
        self.valid_lines = set()
        self._init_hyperparams()
                
        self.reinforce = REINFORCE(1, self.action_dim)
        self.optimizer = Adam(self.reinforce.parameters(), lr=self.learning_rate)
    
    def _init_hyperparams(self):
        num_records = 5
        
        self.gamma = 0
        self.decay_rate = 0.99
        self.entropy_coeff = 0.05
        self.num_episodes = 200000
        self.learning_rate = 0.005
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='Bandit')
        config = wandb.config
        config.gamma = self.gamma
        config.weights = self.weights
        config.decay_rate = self.decay_rate
        config.num_episodes = self.num_episodes
        config.entropy_coeff = self.entropy_coeff
        config.learning_rate = self.learning_rate
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
    
    # Select action according
    def _select_action(self):
        mean, std_dev = self.reinforce([0])
        normal = torch.distributions.Normal(mean, std_dev)
        action = normal.sample()
        return action.numpy()
    
    # Bandit recieves a penalty after every timestep
    def _step(self, problem_instance, action, num_action):
        penalty = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)

        line = CDL.get_lines_from_coeffs(action)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)        
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_action == self.max_lines:
            done = True
            penalty = -super().optimizer(regions, problem_instance)
            self.valid_lines.clear()

        return penalty, done, regions
    
    # Discount penalties for each timestep
    def discount_penalties(self, reward_list):
        reward_array = np.array(reward_list[::-1])
        discounts = (self.gamma ** np.arange(len(reward_array)))
        discounted_returns = np.cumsum(reward_array * discounts)
        discounted_returns = discounted_returns[::-1].reshape(-1, 1).copy()
        G = torch.tensor(discounted_returns, dtype=torch.float32)
        return G
        
    def _learn(self, actions, penalty):
        context_list = np.full(len(actions), 0).reshape(-1, 1)
        action_list = torch.FloatTensor(np.array(actions).reshape(-1, self.action_dim))
        distributed_penalty = penalty / len(actions)
        penalty_list = np.full(len(actions), distributed_penalty)

        self.optimizer.zero_grad()
        discounted_return = self.discount_penalties(penalty_list)
        mean, std_dev = self.reinforce(context_list)
        normal = torch.distributions.Normal(mean, std_dev)
        log_prob = normal.log_prob(action_list)
        loss = -torch.mean(log_prob * discounted_return) - self.entropy_coeff * normal.entropy()
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.entropy_coeff *= self.decay_rate

    # Train bandit on a given problem instance
    def _train(self, problem_instance):        
        for episode in range(self.num_episodes):
            actions = []
            done = False
            num_action = 1
            while not done:
                action = self._select_action()
                actions.append(action)
                penalty, done, regions = self._step(problem_instance, action, num_action)
                num_action += 1
            
            self._learn(actions, penalty)
            wandb.log({"penalty": penalty})
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
                action = self._select_action()
                reward, done, regions = self._step(problem_instance, action, num_action)
                optim_coeffs.append(action)
                num_action += 1
        
        self._log_regions(problem_instance, 'final', regions, reward)
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)
        wandb.finish()
        return optim_coeffs