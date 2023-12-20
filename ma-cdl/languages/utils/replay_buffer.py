import torch
import random

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, state, action, reward, next_state, done):        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        idxs = random.sample(range(self.real_size), batch_size)
        return idxs
    

class CommutativeReplayBuffer(ReplayBuffer):
    def __init__(self, state_size, action_size, buffer_size, max_action):
        super().__init__(state_size, action_size, buffer_size)
        
        self.action_seq = torch.empty(buffer_size, max_action, dtype=torch.int64)
        self.prev_action_seq = torch.empty(buffer_size, max_action, dtype=torch.int64)
        self.prev_action = torch.empty(buffer_size, action_size, dtype=torch.int64)
        self.prev_reward = torch.empty(buffer_size, dtype=torch.float)

    def add(self, transition):
        state, action_seq, action, reward, next_state, done, prev_action_seq, prev_action, prev_reward = transition
        
        super().add(state, action, reward, next_state, done)
        
        self.action_seq[self.count] = torch.as_tensor(action_seq)
        
        if prev_action is not None:
            self.prev_action_seq[self.count] = torch.as_tensor(prev_action_seq)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
        
    def sample(self, batch_size):
        idxs = random.sample(range(self.real_size), batch_size)
        return idxs