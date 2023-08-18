import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class Autoencoder(nn.Module):
    def __init__(self, output_dims):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, output_dims)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, state):
        encoded = self.encoder(state)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoded(self, state):
        with torch.no_grad():
            encoded = self.encoder(state)
        return encoded.flatten().numpy()
    

class DuelingDQN(nn.Module):
    def __init__(self, input_dims, output_dims, lr):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Advantage stream
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128, output_dims)   
        
        # Value stream
        self.val_fc = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
                
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        return val + adv - adv.mean()
    
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 32)
        self.l6 = nn.Linear(32, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1 architecture
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        # Q2 architecture
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    # More efficient to only compute Q1
    def get_Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1