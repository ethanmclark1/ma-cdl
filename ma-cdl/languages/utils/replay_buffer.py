"""
This code is based on the following repository:

Author: Scott Fujimoto
Repository: TD3
URL: https://github.com/sfujim/TD3/blob/master/utils.py
Version: 6a9f761
License: MIT License
"""

import torch
import numpy as np

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.not_done = np.zeros((max_size, 1))


	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]),
			torch.FloatTensor(self.action[ind]),
			torch.FloatTensor(self.reward[ind]),
			torch.FloatTensor(self.next_state[ind]),
			torch.FloatTensor(self.not_done[ind])
		)