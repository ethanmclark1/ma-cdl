import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Planner(nn.Module):
    def __init__(self, input_dums, output_dims):
        super(Planner, self).__init__()
        self.linear_1 = nn.Linear(input_dums, 64)
        self.linear_2 = nn.Linear(64, 32)
        self.linear_3 = nn.Linear(32, output_dims)

    def forward(self, start, goal, obstacles, env_shape):
        start = torch.tensor(start, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        obstacles = torch.tensor(obstacles, dtype=torch.float).flatten()
        env_shape = torch.tensor([env_shape], dtype=torch.float)

        state = torch.cat((start, goal, obstacles, env_shape), dim=-1)
        activation1 = F.relu(self.linear_1(state))
        activation2 = F.relu(self.linear_2(activation1))
        output = self.linear_3(activation2)
        return output
    
class LanguageSync(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LanguageSync, self).__init__()
        self.linear_1 = nn.Linear(input_dims, 16)
        self.linear_2 = nn.Linear(16, 32)
        self.linear_3 = nn.Linear(32, output_dims)
            
    def forward(self, env_type, num_obstacles):
        env_type = torch.tensor([env_type], dtype=torch.float)
        num_obstacles = torch.tensor([num_obstacles], dtype=torch.float)
        state = torch.cat((env_type, num_obstacles), dim=-1)
        activation1 = F.relu(self.linear_1(state))
        activation2 = F.relu(self.linear_2(activation1))
        output = self.linear_3(activation2)
        return output

class ListenerNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ListenerNet, self).__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=1, 
            num_layers=2, batch_first=True
            )
        self.linear_1 = nn.Linear(input_dims, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, output_dims)

    def forward(self, obs, x):
        obs = torch.tensor(obs)
        x = torch.tensor(x, dtype=torch.float)
        x = torch.unsqueeze(x, dim=-1)
        sequence, _ = self.gru(x)
        sequence = torch.squeeze(sequence)
        
        state = torch.cat((obs, sequence), dim=-1)
        activation1 = F.relu(self.linear_1(state))
        activation2 = F.relu(self.linear_2(activation1))
        output = self.linear_3(activation2)
        return output
    
