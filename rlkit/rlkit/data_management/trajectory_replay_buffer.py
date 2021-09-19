"""
A replay buffer that associates trajectories together.

Author: Ian Char
Date: 4/7/2021
"""
from collections import OrderedDict

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class PPOBuffer(object):

    def __init__(
        self,
        max_num_trajectories,
        max_trajectory_length,
        observation_dim,
        action_dim,
        env_info_sizes,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_num_trajectories = max_num_trajectories
        self._max_trajectory_length = max_trajectory_length
        
