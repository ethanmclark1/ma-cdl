import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

torch.manual_seed(42)


class RewardEstimator(nn.Module):
    def __init__(self, input_dims, lr, step_size, gamma, dropout_rate):
        super(RewardEstimator, self).__init__()
        
        fc1_output = 128 if input_dims == 63 else 64
        
        self.fc1 = nn.Linear(in_features=input_dims, out_features=fc1_output)   
        self.ln1 = nn.LayerNorm(fc1_output)
        
        self.fc2 = nn.Linear(in_features=fc1_output, out_features=32)     
        self.ln2 = nn.LayerNorm(32)
        
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.optim = Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=gamma)
        
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
    

class DQN(nn.Module):
    def __init__(self, state_dims, action_dim, lr):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dims, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))
        a = torch.tanh(self.fc4(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        # Q2 architecture
        self.fc5 = nn.Linear(state_dim + action_dim, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1 architecture
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.fc4(x1)

        # Q2 architecture
        x2 = F.relu(self.fc5(xu))
        x2 = F.relu(self.fc6(x2))
        x2 = F.relu(self.fc7(x2))
        x2 = self.fc8(x2)
        return x1, x2

    # More efficient to only compute Q1
    def get_Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.fc4(x1)
        return x1