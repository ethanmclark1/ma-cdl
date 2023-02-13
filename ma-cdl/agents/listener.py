import pdb
import torch
import numpy as np

from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR

from agents.utils.networks import ListenerNet

class Listener():
    def __init__(self, input_dims=64, output_dims=4):
        self._init_hyperparams()
        self.actor = ListenerNet(input_dims, output_dims)
        self.critic = ListenerNet(input_dims, 1)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.actor_scheduler = ExponentialLR(self.actor_optim, gamma=self.gamma)
        self.critic_scheduler = ExponentialLR(self.critic_optim, gamma=self.gamma)
        
    def _init_hyperparams(self):
        self.lr = 10e-3
        self.clip = 0.2
        self.gamma = 0.99
        self.n_updates_per_iteration = 5
                        
    def get_action(self, obs, directions, setting):
        logits = self.actor(obs, directions)
        action_probs = softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample() if setting == 'train' else dist.probs.argmax()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    # Gather value function for critic to critique actor with & log probs for actor to calculate loss with
    def evaluate(self, batch_obs, batch_messages, batch_actions):
        V = self.critic(batch_obs, batch_messages).squeeze()
        logits = self.actor(batch_obs, batch_messages)
        action_probs = softmax(V, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs
    
    def train(self, batch_trajectory):
        actor_loss_data, critic_loss_data = [], []
        batch_messages, batch_obs, batch_actions, batch_log_probs, batch_rtgs = batch_trajectory
        
        V, _ = self.evaluate(batch_obs, batch_messages, batch_actions)
        advantage = batch_rtgs - V.detach()
        # Normalized advantage reduces the variance
        normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        for _ in range(self.n_updates_per_iteration):
            # Calculate pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_messages, batch_actions)
            # Calculate ratio
            ratio = torch.exp(curr_log_probs - batch_log_probs)
            # Calculate surrogate losses
            surrogate_1 = ratio * normalized_advantage
            surrogate_2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * normalized_advantage

            actor_loss = (-torch.min(surrogate_1, surrogate_2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            actor_loss_data.append(actor_loss.item())
            
            critic_loss = MSELoss()(V, batch_rtgs)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            critic_loss_data.append(critic_loss.item())
            
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return np.mean(actor_loss_data), np.mean(critic_loss_data)
            
    # Calculate discounted return for each episode
    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs