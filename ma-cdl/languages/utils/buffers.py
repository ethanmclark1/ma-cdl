import torch

# Used for RL Agents
class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, rng):
        action_dtype = torch.int64 if action_size == 1 else torch.float
        
        self.state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.next_state = torch.zeros(buffer_size, state_size, dtype=torch.float)
        self.done = torch.zeros(buffer_size, dtype=torch.bool)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.rng = rng
        # Managing buffer size and current position
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
        idxs = self.rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class CommutativeReplayBuffer(ReplayBuffer):
    def __init__(self, state_size, action_size, buffer_size, max_action, rng):
        super().__init__(state_size, action_size, buffer_size, rng)

        action_dtype = self.action.dtype
        prev_state = max_action if action_dtype == torch.int64 else max_action*action_size
        
        self.prev_state = torch.zeros(buffer_size, prev_state, dtype=torch.float)
        self.prev_action = torch.zeros(buffer_size, action_size, dtype=action_dtype)
        self.prev_reward = torch.zeros(buffer_size, dtype=torch.float)
        self.has_previous = torch.zeros(buffer_size, dtype=torch.bool)

    def add(self, state, action, reward, next_state, done, prev_state, prev_action, prev_reward):        
        super().add(state, action, reward, next_state, done, increase_size=False)
                
        if prev_state is not None:
            self.prev_state[self.count] = torch.as_tensor(prev_state)
            self.prev_action[self.count] = torch.as_tensor(prev_action)
            self.prev_reward[self.count] = torch.as_tensor(prev_reward)
            self.has_previous[self.count] = True
            
        self._increase_size()
        

# Used for Reward Estimator
class RewardBuffer:
    def __init__(self, buffer_size, step_dims, rng):       
        self.transition = torch.zeros(buffer_size, step_dims, dtype=torch.float) 
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.rng = rng
        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self):
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, state, action, reward, next_state):     
        if isinstance(action, int):
            action = [action]
            
        state = torch.as_tensor(state)
        action = torch.as_tensor(action)
        next_state = torch.as_tensor(next_state)   
        
        self.transition[self.count] = torch.cat((state, action, next_state), dim=0)
        self.reward[self.count] = torch.tensor((reward))
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size):
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = self.rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class CommutativeRewardBuffer(RewardBuffer):
    def __init__(self, buffer_size, step_dims, rng):
        super().__init__(buffer_size, step_dims, rng)
        self.transition = torch.zeros((buffer_size,2,step_dims), dtype=torch.float)
        self.reward = torch.zeros(buffer_size, 2, dtype=torch.float)
        
    def add(self, prev_state, action, prev_reward, commutative_state, prev_action, reward, next_state):
        if isinstance(action, int):
            action = [action]
            prev_action = [prev_action]
        
        prev_state = torch.as_tensor(prev_state)
        prev_action = torch.as_tensor(prev_action)
        commutative_state = torch.as_tensor(commutative_state)
        action = torch.as_tensor(action)
        next_state = torch.as_tensor(next_state)  
        
        step_0 = torch.cat((prev_state, action, commutative_state), dim=0)
        step_1 = torch.cat((commutative_state, prev_action, next_state), dim=0)
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.tensor((prev_reward, reward))
        self.is_initialized[self.count] = True

        self._increase_size()