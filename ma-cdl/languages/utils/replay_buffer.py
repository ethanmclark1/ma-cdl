"""
This code is based on the following repository:

Author: OpenAI
Repository: Baselines
URL: https://github.com/openai/baselines
Version: ea25b9e
License: MIT License
"""

import numpy as np
import random
from itertools import permutations

class ReplayBuffer(object):
    def __init__(self, size=750000):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    # Shuffle the action order when adding to the replay buffer
    def add(self, obs_t, action, reward):
        action_shuffle = np.array(list(permutations(action.reshape(-1, 3))))
        
        for shuffled_action in action_shuffle:
            action = shuffled_action.reshape(1, -1)                      
            data = (obs_t, action, reward)

            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
        return np.array(obses_t), np.array(actions), np.array(rewards)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
