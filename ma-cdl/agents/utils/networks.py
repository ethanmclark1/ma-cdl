import torch.nn as nn
import torch.nn.functional as F

# RNN for speaker, given start, goal, obstacles, and prior directions
# Output is a region label
class SpeakerNetwork:
    def __init__(self):
        a=3

class ListenerNetwork:
    def __init__(self, obs_dim):
        super(ListenerNetwork, self).__init__()
        
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 4)
    
    def forward(self, x):
        a=3