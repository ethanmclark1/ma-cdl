import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    def __init__(self, input_dims, output_dims, learning_rate):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Advantage stream
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, output_dims)   
        
        # Value stream
        self.val_fc = nn.Linear(128, 32)
        self.val_out = nn.Linear(32, 1)
        
        self.optim = Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.90, patience=200)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        return val + adv - adv.mean()