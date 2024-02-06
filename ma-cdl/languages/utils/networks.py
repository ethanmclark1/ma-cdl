import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


torch.manual_seed(42)

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


class RewardEstimator(nn.Module):
    def __init__(self, input_dims, lr, step_size, gamma):
        super(RewardEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.ln1 = nn.LayerNorm(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        
        self.fc3 = nn.Linear(64, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=gamma)
        
    def forward(self, x):
        x = x.float()
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
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
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        # Q2 architecture
        self.fc5 = nn.Linear(state_dim + action_dim, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 1)
        
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