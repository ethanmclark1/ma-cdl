import copy
import torch

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import DuelingDQN
from languages.utils.replay_buffer import PrioritizedReplayBuffer

"""Using Dueling DDQN with Prioritized Experience Replay"""
class Discrete(CDL):
    def __init__(self):
        super().__init__()        
        self._init_hyperparams()    
        
        self.num_actions = 2    
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 5e-3
        self.alpha = 1e-4
        self.gamma = 0.99
        self.batch_size = 256
        self.granularity = 0.20
        self.memory_size = 30000
        self.epsilon_start = 1.0
        self.dummy_episodes = 200
        self.num_episodes = 15000
        self.epsilon_decay = 0.9997
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
        config.granularity = self.granularity
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dummy_episodes = self.dummy_episodes
        
    def _decrement_exploration(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.01, self.epsilon)
        
    # Select a line for a given state based on epsilon-greedy policy
    def _select_action(self, state):
        with torch.no_grad():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.num_actions)
            else:
                q_vals = self.dqn(torch.tensor(state))
                action = torch.argmax(q_vals).item()
        
        return action
            
    def _step(self, problem_instance, action, num_lines):
        line = self.candidate_lines[action]
        reward, next_state, done, _ = super()._step(problem_instance, line, num_lines)
        return reward, next_state, done, _
    
    # Learn from the replay buffer following the DDQN algorithm
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        q_values = self.dqn(state)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.dqn(next_state)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).view(-1).detach()
        
        q_hat = reward + (1 - done) * self.gamma * next_q
        
        td_error = torch.abs(q_hat - q).detach()
        
        if weights is None:
            weights = torch.ones_like(q)

        self.dqn.optim.zero_grad()
        # Multiply by importance sampling weights to correct bias from prioritized replay
        loss = (weights * (q_hat - q) ** 2).mean()
        loss.backward()
        self.dqn.optim.step()
                
        # Update target network
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return None, loss.item(), td_error.numpy(), tree_idxs
    
    def _generate_optimal_lines(self):        
        # Start from a blank slate every time
        self.epsilon = self.epsilon_start
        self.buffer = PrioritizedReplayBuffer(self.state_dim, 1, self.memory_size)
        self.dqn = DuelingDQN(self.state_dim, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        optim_lines = super()._generate_optimal_lines()
          
        return optim_lines  