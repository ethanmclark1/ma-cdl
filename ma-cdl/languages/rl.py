import time
import wandb
import torch
import random
import itertools
import numpy as np

from torch.optim import Adam
from languages.utils.ae import AE
from languages.utils.cdl import CDL
from languages.utils.networks import DuelingDQN
from languages.utils.replay_buffer import ReplayBuffer

""" Using Dueling Deep Q-Network (DuelingDQN) """
class RL(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self.states = []
        self.actions = []
        self.reward = []
        self.next_states = []
        self.dones = []
        
        self.valid_lines = set()
        
        self.action_dim = 3
        self.state_dim = 128
        self.loss = torch.nn.MSELoss()
        self.action_grid = np.arange(-1, 1 + self.resolution, self.resolution)
        
        self._init_hyperparams()
        self.autoencoder = AE(self.state_dim, self.rng, self.max_lines, self.action_grid)
        self.network = DuelingDQN(self.state_dim, self.action_dim)
        self.target = DuelingDQN(self.state_dim, self.action_dim)
        self.optim = Adam(self.network.parameters(), lr=self.alpha)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        
    def _init_hyperparams(self):
        num_records = 10
        num_network_updates = 20
        
        self.alpha = 3e-4
        self.gamma = 0.99
        self.dummy_eps = 200
        self.batch_size = 256
        self.epsilon_start = 1 
        self.min_epsilon = 0.05
        self.num_episodes = 2000
        self.num_iterations = 100
        self.epsilon_decay = 0.99 
        self.record_freq = self.num_episodes // num_records
        self.replace_freq = self.num_episodes // num_network_updates
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='RL')
        config = wandb.config
        config.weights = self.weights
        config.resolution = self.resolution
        config.configs_to_consider = self.configs_to_consider
        
        # Hyperparameters
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.dummy_eps = self.dummy_eps
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.num_iterations = self.num_iterations
        
        # Neural Network
        config.l1 = self.network.l1
        config.l2 = self.network.l2
        config.value_layer = self.network.value
        config.advantage_layer = self.network.advantage
    
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, episode, regions, reward):
        pil_image = super()._get_image(problem_instance, 'Episode', episode, regions, reward)
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
            
        next_state = self.autoencoder.get_state(regions)
        return reward, next_state, done, regions
    
    # Generate samples of permuted actions to form the basis of transition hallucinations
    def _hallucinate(self):
        if len(self.actions) > 4:
            actions = self.actions.copy()
            shuffled_actions = []
            for _ in range(24):
                random.shuffle(actions)
                shuffled_actions.append(actions.copy())
        else:
            shuffled_actions = list(itertools.permutations(self.actions))

        return shuffled_actions
    
    # Add transition to replay buffer
    def _remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.states.append(state)
        self.actions.append(action)
        self.reward.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:            
            default_boundary_lines = CDL.get_valid_lines([])  
            default_square = CDL.create_regions(default_boundary_lines)
            start_state = self.autoencoder.get_state(default_square)
            
            shuffled_actions = self._hallucinate()
            for shuffled_action in shuffled_actions:
                state = start_state
                self.valid_lines.clear()
                self.valid_lines.update(default_boundary_lines)
                
                for idx, action in enumerate(shuffled_action):
                    line = CDL.get_lines_from_coeffs(action)
                    self.valid_lines.update(CDL.get_valid_lines(line))
                    regions = CDL.create_regions(list(self.valid_lines))
                    next_state = self.autoencoder.get_state(regions)
                    reward = self.reward[idx]
                    done = True if idx == len(shuffled_action) - 1 else False
                    
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
            
            self.states = []
            self.actions = []
            self.reward = []
            self.next_states = []
            self.dones = []
            self.valid_lines.clear()
            
    # Populate replay buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_eps):
            done = False
            num_action = 1
            state = start_state
            while not done:
                action = self._select_action(state)
                reward, next_state, done, _ = self._step(problem_instance, action, num_action)
                self._remember(state, action, reward, next_state, done)
                state = next_state
                num_action += 1
                                
    # Select an action (coefficients of a linear line)
    def _select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_grid, size=self.action_dim)
            action = np.digitize(action, self.action_grid)
        else:
            q_values = self.network(state)
            action_idx = torch.argmax(q_values).detach().numpy()
            action = self.action_grid[action_idx]
        return action
    
    # Update the target network
    def _update_target(self, episode):
        if episode > 0 and episode % self.replace_freq == 0:
            self.target.load_state_dict(self.network.state_dict())

    # Learn from the replay buffer
    def _learn(self, episode):
        self._update_target(episode)
        
        for _ in range(self.num_iterations):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            network_qval = self.network(states)
            target_qval = rewards + (1 - dones) * self.gamma * self.target(next_states).detach().max(1)[0].unsqueeze(1)
            
            self.optim.zero_grad()
            loss = self.loss(target_qval, network_qval)
            loss.backward()
            self.optim.step()
            
    # Decrease rate of exploratio as time goes on
    def _decrement_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
    
    # Train model on a given problem_instance
    def _train(self, problem_instance, start_state):        
        returns = []     
        
        for episode in range(self.num_episodes):
            done = False
            num_action = 0
            state = start_state
            while not done: 
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done, regions = self._step(problem_instance, action, num_action)  
                self._remember(state, action, reward, next_state, done)         
                self._learn()
                self._decrement_epsilon()
                state = next_state
                
            returns.append(reward)
            avg_returns = np.mean(returns[-self.replace_freq:])
            wandb.log({"Average Returns": avg_returns})
            if episode % self.record_freq == 0 and len(regions) > 1:
                self._log_regions(problem_instance, episode, regions, reward)
    
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        start_time = time.time()
        
        valid_lines = CDL.get_valid_lines([])   
        default_square = CDL.create_regions(valid_lines)
        start_state = self.autoencoder.get_state(default_square)
        
        self.epsilon = self.epsilon_start
        self._populate_buffer(problem_instance, start_state)
        self._train(problem_instance, start_state)
        
        done = False
        optim_coeffs = []
        self.epsilon = 0
        with torch.no_grad():
            num_action = 1
            state = start_state
            while not done: 
                action = self._select_action(state)
                optim_coeffs.append(action)
                reward, next_state, done, regions = self._step(problem_instance, action, num_action)
                state = next_state
                num_action += 1
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
        wandb.log({"Elapsed Time": elapsed_time})
        
        self._log_regions(problem_instance, 'Final', regions, reward)
        wandb.finish()  
        optim_coeffs = np.array(optim_coeffs).reshape(-1, self.action_dim)   
        return optim_coeffs  