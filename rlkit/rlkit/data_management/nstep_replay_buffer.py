from collections import OrderedDict

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class NStepReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        n_steps,
        observation_dim=None,
        action_dim=None,
        env=None,
        env_info_sizes=None,
    ):
        if env is not None:
            self._observation_dim = env.observation_space.low.size
            self._action_dim = env.action_space.low.size
        elif observation_dim is not None and action_dim is not None:
            self._observation_dim = observation_dim
            self._action_dim = action_dim
        else:
            raise ValueError('Need observation and action dimensions.')
        self._max_replay_buffer_size = max_replay_buffer_size
        self._nsteps = n_steps
        if env_info_sizes is None:
            if env is not None and hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        self._observations = np.zeros((max_replay_buffer_size,
                                       observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, n_steps,
                                   observation_dim))
        self._steps = np.zeros((max_replay_buffer_size, 1))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, n_steps))
        # self._terminals[i] = a terminal was received at time i + nstep
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        raise NotImplementedError('Cannot add single sample in Nstep Buffer')

    def add_path(self, path):
        path_len = len(path['observations'])
        for t in range(path_len):
            nstep = min(path_len - t, self._nsteps)
            self._observations[self._top] = path['observations'][t]
            self._actions[self._top] = path['actions'][t]
            self._rewards[self._top, :nstep] \
                    = path['rewards'][t: t + nstep].flatten()
            self._terminals[self._top] = path['terminals'][t + nstep - 1]
            self._next_obs[self._top, :nstep] =\
                    path['next_observations'][t:t + nstep]
            self._steps[self._top] = nstep
            if 'env_info' in path:
                for key in self._env_info_keys:
                    self._env_infos[key][self._top] = path['env_info'][key]
            self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            steps=self._steps[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
