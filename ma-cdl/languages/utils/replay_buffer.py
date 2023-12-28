import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.next_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.done = torch.zeros(buffer_size, dtype=torch.int64)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.int64)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self):
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, state, action, reward, next_state, done, increase_size=True):        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.is_initialized[self.count] = True

        if increase_size:
            self._increase_size()

    def sample(self, batch_size):
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = np.random.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class CommutativeReplayBuffer(ReplayBuffer):
    def __init__(self, state_size, action_size, buffer_size, max_action):
        super().__init__(state_size, action_size, buffer_size)
        
        self.prev_state_proxy = torch.zeros(buffer_size, max_action, dtype=torch.int64)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=torch.int64)
        self.prev_reward = torch.zeros(buffer_size, dtype=torch.float)
        self.has_previous = torch.zeros(buffer_size, dtype=torch.int64)

    def add(self, transition):
        state, action, reward, next_state, done, prev_state_proxy, prev_action, prev_reward = transition
        
        super().add(state, action, reward, next_state, done, increase_size=False)
                
        if prev_action is not None:
            self.prev_state_proxy[self.count] = torch.as_tensor(prev_state_proxy)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
            self.has_previous[self.count] = True
            
        self._increase_size()