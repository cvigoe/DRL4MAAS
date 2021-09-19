"""
Utility for the offline setting.
"""
from collections import OrderedDict

import numpy as np

from rlkit.data_management.nstep_replay_buffer import NStepReplayBuffer


def qlearning_dataset_to_nstep_replay_buffer(dataset, nsteps):
    """Turn qlearning dataset into nstep replay buffer.
    Note: It is assumed that the qlearning dataset is sorted.
    Args:
        dataset: Dictionary of observations, actions, etc. mapping to an ndarry
    """
    rbuffer = NStepReplayBuffer(
        max_replay_buffer_size=len(dataset['observations']),
        observation_dim=dataset['observations'].shape[1],
        action_dim=dataset['actions'].shape[1],
        n_steps=nsteps,
    )
    bookends = np.append(
        0,
        np.argwhere(
            np.all(
                dataset['observations'][1:]
                != dataset['next_observations'][:-1], axis=1)
        ) + 1)
    bookends = np.append(bookends, len(dataset['observations']))
    for idx in range(len(bookends) - 1):
        start, end = bookends[idx], bookends[idx + 1]
        rbuffer.add_path(OrderedDict(
            observations=dataset['observations'][start:end],
            actions=dataset['actions'][start:end],
            rewards=dataset['rewards'][start:end],
            next_observations=dataset['next_observations'][start:end],
            terminals=dataset['terminals'][start:end],
        ))
    return rbuffer
