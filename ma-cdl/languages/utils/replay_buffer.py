import torch
import random

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, max_action):
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.idx = torch.empty(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)
        self.index_seq = torch.empty(buffer_size, max_action, dtype=torch.int64)
        
        self.prev_index_seq = torch.empty(buffer_size, max_action, dtype=torch.int64)
        self.prev_index = torch.empty(buffer_size, action_size, dtype=torch.int64)
        self.prev_reward = torch.empty(buffer_size, dtype=torch.float)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.target_size = max_action

    def add(self, transition):        
        state, idx, reward, next_state, done, index_seq, prev_index_seq, prev_index, prev_reward = transition

        self.state[self.count] = torch.as_tensor(state)
        self.idx[self.count] = torch.as_tensor(idx)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.index_seq[self.count] = torch.as_tensor(index_seq)
        
        if prev_index is not None:
            self.prev_index_seq[self.count] = torch.as_tensor(prev_index_seq)
            self.prev_index[self.count] = torch.as_tensor(prev_index)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    # Sample batch of transitions with importance sampling weights
    def sample(self, batch_size):
        # Uniformly sample a batch of experiences
        idxs = random.sample(range(self.real_size), batch_size)
        
        batch = (
            self.state[idxs],
            self.idx[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs],
            self.prev_index_seq[idxs],
            self.prev_index[idxs],
            self.prev_reward[idxs],
            self.index_seq[idxs]
        )

        return batch