"""
Class for storing data.
"""
from collections import OrderedDict

import numpy as np


class OfflineDataStore(object):

    def __init__(self, data):
        """Constructor.
        Args:
            data: The offline dataset. Takes data structures...
                * tuple: List of tuples (s, a, r, next_s, done)
                * dict: Dictionary with keys mapping to observation, etc
        """
        if type(data) is dict or type(data) is OrderedDict:
            self._observations = data['observations']
            self._actions = data['actions']
            self._rewards = data['rewards']
            self._next_obs = data['next_observations']
            self._terminals = data['terminals']
            self._size = self._observations.shape[0]
        else:
            self._size = len(data)
            tup_elements = [np.asarray(te) for te in zip(*data)]
            self._observations = tup_elements[0]
            self._actions = tup_elements[1]
            if len(self._actions.shape) == 1:
                self._actions = self._actions.reshape(-1, 1)
            self._rewards = tup_elements[2].reshape(-1, 1)
            self._next_obs = tup_elements[3]
            self._terminals = tup_elements[4].reshape(-1, 1)

    def random_batch(self, batch_size):
        """Get a batch of the data."""
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices].reshape(-1, 1),
            terminals=self._terminals[indices].reshape(-1, 1),
            next_observations=self._next_obs[indices],
        )
        return batch
