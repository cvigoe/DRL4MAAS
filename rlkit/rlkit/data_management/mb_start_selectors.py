"""
Methods for collecting starts for model based methods.

Author: Ian Char
"""
import abc
from typing import Optional

import numpy as np

from rlkit.data_management.offline_data_store import OfflineDataStore

class StartSelector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_starts(self, num_starts: int) -> np.ndarray:
        """Get start states."""

class DataStartStateSelector(StartSelector):

    def __init__(
            self,
            data: OfflineDataStore,
            true_starts: Optional[np.ndarray] = None,
            prop_true: float = 0,
    ):
        """Constructor.
        Args:
            data: The data.
            true_starts: States marked as starts in the dataset.
            prop_true: The proportion from the true start states to draw from.
        """
        self._data = data
        self._true_starts = true_starts
        self._prop_true = prop_true

    def get_starts(self, num_starts: int) -> np.ndarray:
        """Get start states."""
        from_starts = int(self._prop_true * num_starts)
        if self._true_starts is None or from_starts == 0:
            return self._data.random_batch(num_starts)['observations']
        from_batch = num_starts - from_starts
        startidxs = np.random.choice(len(self._true_starts), size=from_starts)
        start_starts = self._true_starts[startidxs]
        batch_starts = self.offline_data.random_batch(from_batch)['observations']
        return np.vstack([start_starts, batch_starts])

class UniformStartStateSelector(StartSelector):

    def __init__(
        self,
        low_bounds: np.ndarray,
        high_bounds: np.ndarray,
    ):
        self._low_bounds = np.array(low_bounds)
        self._high_bounds = np.array(high_bounds)


    def get_starts(self, num_starts: int) -> np.ndarray:
        """Get start states."""
        return np.random.uniform(
            self._low_bounds,
            self._high_bounds,
            (num_starts, len(self._low_bounds)),
        )
