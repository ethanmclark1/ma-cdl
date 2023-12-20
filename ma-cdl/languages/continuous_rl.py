import copy
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from languages.utils.ae import AE
from languages.utils.cdl import CDL, SQUARE
from languages.utils.replay_buffer import ReplayBuffer, CommutativeReplayBuffer


class BasicDDPG(CDL):
    def __init__(self, scenario, world):
        super(BasicDDPG, self).__init__(scenario, world)
        
    def _init_hyperparams(self):
        pass
        
    def _init_wandb(self, problem_instance):
        pass
    
    def _select_action(self, state, noise):
        pass
    
    def _update(self, episode, loss):
        pass
    
    def _learn(self, indices):
        pass
    
    def _train(self, problem_instance):
        pass
    
    def _get_final_lines(self, problem_instance):
        pass
    
    def _generate_language(self, problem_instance):
        pass
    

class CommutativeDDPG(BasicDDPG):
    def __init__(self, scenario, world):
        super(CommutativeDDPG, self).__init__(scenario, world)
        
    def _learn(self):
        pass
    
    def _train(self):
        pass
    
    def _generate_language(self, problem_instance):
        pass