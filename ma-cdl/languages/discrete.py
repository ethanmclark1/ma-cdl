import copy
import torch

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import DuelingDQN
from languages.utils.replay_buffer import PrioritizedReplayBuffer

"""Using Dueling DDQN with Prioritized Experience Replay"""
class Discrete(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)     
           
        self._init_hyperparams()        
        self._create_candidate_set_of_lines()
        self.num_actions = len(self.candidate_lines)
        
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines, self.candidate_lines)

    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 3e-3
        self.alpha = 7e-4
        self.gamma = 0.999
        self.batch_size = 512
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
        
    # Generate possible set of lines to choose from
    def _create_candidate_set_of_lines(self):
        self.candidate_lines = []
        granularity = int(self.granularity * 100)
        
        # termination line
        self.candidate_lines += [(0, 0, 0)]
        
        # vertical/horizontal lines
        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0, i)] # vertical lines
            self.candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            self.candidate_lines += [(0.1, 0.1, i)]
            self.candidate_lines += [(-0.1, 0.1, i)]
        
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
        reward, next_state, done, _ = super().step(problem_instance, line, num_lines)
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
    
    def _generate_optimal_lines(self, problem_instance):        
        # Start from a blank slate every time
        self.epsilon = self.epsilon_start
        self.buffer = PrioritizedReplayBuffer(self.state_dim, self.memory_size)
        self.dqn = DuelingDQN(self.state_dim, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        optim_lines = super()._generate_optimal_lines(problem_instance)
          
        return optim_lines  