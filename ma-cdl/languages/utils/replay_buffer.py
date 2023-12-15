import torch
import random


class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):        
        state, action, reward, next_state, done = transition

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    # Sample batch of transitions with importance sampling weights
    def sample(self, batch_size):
        # Uniformly sample a batch of experiences
        idxs = random.sample(range(self.real_size), batch_size)
        
        batch = (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs]
        )

        return batch