import copy
import wandb
import torch
import numpy as np

from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import DuelingDQN
from languages.utils.replay_buffer import PrioritizedReplayBuffer

"""Using a Dueling DDQN with Prioritized Experience Replay"""
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)     
        self._init_hyperparams()
           
        self.valid_lines = set()
        
        self.state_dim = 128
        self.num_actions = len(self.possible_coeffs)
        
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines)
        
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 0.003
        self.alpha = 1e-3
        self.gamma = 0.9998
        self.patience = 300
        self.batch_size = 512
        self.memory_size = 25000
        self.epsilon_start = 1.0
        self.dummy_episodes = 200
        self.num_episodes = 20000
        self.epsilon_decay = 0.9996
        self.record_freq = self.num_episodes // num_records
            
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'RL/{problem_instance.capitalize()}')
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dummy_episodes = self.dummy_episodes
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
        wandb.log({"image": wandb.Image(pil_image)})
        
    def _decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.01, self.epsilon)
        
    # Select coefficients for a given state
    def _select_action(self, state):
        with torch.no_grad():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.num_actions)
            else:
                q_vals = self.dqn(torch.tensor(state))
                action = torch.argmax(q_vals).item()
        
        return action
    
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_episodes):
            done = False
            num_lines = 0
            state = start_state
            while not done:
                num_lines += 1
                action = self._select_action(state)
                reward, next_state, done, _ = self._step(problem_instance, action, num_lines)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
            
    # Overlay lines in the environment
    def _step(self, problem_instance, action, num_lines):
        reward = 0
        done = False
        prev_num_lines = max(len(self.valid_lines), 4)
        
        coeffs = self.possible_coeffs[action]
        line = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(line)
        self.valid_lines.update(valid_lines)
        regions = CDL.create_regions(list(self.valid_lines))
        
        if len(self.valid_lines) == prev_num_lines or num_lines == self.max_lines:
            done = True
            reward = super().optimizer(regions, problem_instance)
            self.valid_lines.clear()
        
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
    
    # Learn from the replay buffer
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
        self.dqn.scheduler.step(loss.item())
                
        # Update target network
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item(), td_error.numpy(), tree_idxs
    
    # Train model on a given problem_instance
    def _train(self, problem_instance, start_state):  
        losses = []      
        returns = []     
        best_coeffs = None
        best_regions = None
        best_reward = -np.inf

        for episode in range(self.num_episodes):
            done = False
            num_lines = 0
            action_lst = []
            state = start_state
            while not done: 
                num_lines += 1
                action = self._select_action(state)                
                reward, next_state, done, regions = self._step(problem_instance, action, num_lines)  
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
                action_lst.append(action)
            
            loss, td_error, tree_idxs = self._learn()
            
            self.buffer.update_priorities(tree_idxs, td_error)
            self._decay_epsilon()
            
            losses.append(loss)
            returns.append(reward)
            avg_losses = np.mean(losses[-100:])
            avg_returns = np.mean(returns[-100:])
            wandb.log({"Average Loss": avg_losses})
            wandb.log({"Average Returns": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
                
            if reward > best_reward:
                best_coeffs = list(map(lambda action: self.possible_coeffs[action], action_lst))
                best_regions = regions
                best_reward = reward
        
        return best_coeffs, best_regions, best_reward
    
    def _generate_optimal_coeffs(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        self.buffer = PrioritizedReplayBuffer(self.state_dim, self.memory_size)
        self.dqn = DuelingDQN(self.state_dim, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self._init_wandb(problem_instance)
        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        self._populate_buffer(problem_instance, start_state)
        best_coeffs, best_regions, best_reward = self._train(problem_instance, start_state)
                                
        self._log_regions(problem_instance, 'Final', best_regions, best_reward)
        wandb.log({"Final Reward": best_reward})
        wandb.finish()  
        
        optim_coeffs = np.array(best_coeffs).reshape(-1, 3)   
        return optim_coeffs  