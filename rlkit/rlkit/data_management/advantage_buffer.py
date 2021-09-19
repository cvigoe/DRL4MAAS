"""
Buffer that also stores advantage estimates as described in PPO.

Author: Ian Char
Date: 4/7/2021
"""
import numpy as np
import torch
import pudb
import copy
from rlkit.data_management.onpolicy_buffer import OnPolicyBuffer
from rlkit.torch.core import torch_ify


class AdvantageReplayBuffer(OnPolicyBuffer):
    """Buffer that also stores advantages."""

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        value_function,
        intrinsic_value_function,
        tdlambda=0.95,
        discount=0.99,
        intrinsic_discount = 0.9999,
        target_lookahead=10,
        use_dones_for_rnd_critic=False,
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = env.observation_space.low.size
        self._action_dim = env.action_space.low.size
        self._value_function = value_function
        self._intrinsic_value_function = intrinsic_value_function
        self._tdlambda = tdlambda
        self._discount = discount
        self._intrinsic_discount = intrinsic_discount
        self.use_dones_for_rnd_critic = use_dones_for_rnd_critic
        self._target_lookahead = target_lookahead
        self.clear_buffer()

    def clear_buffer(self):
        """Clear the replay buffers."""
        # Set up buffers.
        self._observations = np.zeros((self._max_replay_buffer_size,
                                       self._observation_dim))
        self._next_obs = np.zeros((self._max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((self._max_replay_buffer_size,
                                  self._action_dim))
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        self._intrinsic_rewards = np.zeros((self._max_replay_buffer_size, 1))
        self._intrinsic_rewards_raw = np.zeros((self._max_replay_buffer_size, 1))
        self._advantages = np.zeros((self._max_replay_buffer_size, 1))
        self._intrinsic_advantages = np.zeros((self._max_replay_buffer_size, 1))
        self._targets = np.zeros((self._max_replay_buffer_size, 1))
        self._intrinsic_targets = np.zeros((self._max_replay_buffer_size, 1))
        self._logpis = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1),
                                   dtype='uint8')
        self._top = 0
        self._size = 0
        self.running_std_intrinsic_return = 0
        self.running_var_intrinsic_return = 0
        self.running_mean_intrinsic_return = 0
        self.num_return_samples = 0

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            intrinsic_rewards=self._intrinsic_rewards[indices],
            intrinsic_rewards_raw=self._intrinsic_rewards_raw[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            advantages=self._advantages[indices],
            intrinsic_advantages=self._intrinsic_advantages[indices],
            targets=self._targets[indices],
            intrinsic_targets=self._intrinsic_targets[indices],
            logpis=self._logpis[indices],
        )
        return batch

    def add_path(self, path):
        obs = path['observations']
        acts = path['actions']
        rews = path['rewards']
        intrinsic_rews = path['intrinsic_rewards']
        intrinsic_rews_raw = copy.deepcopy(intrinsic_rews)
        nxts = path['next_observations']
        dones = path['terminals']
        pis = np.array([ai['logpi'] for ai in path['agent_infos']])
        path_len = len(obs)


        # Normalise the intrinsic rewards
        # NOTE: this is incorrect, it should be normalizing by a running std. dev. of the intrinsic returns, not the rewards
        self.num_return_samples += 1
        self.running_mean_intrinsic_return += (1/self.num_return_samples) * (np.sum(intrinsic_rews) - self.running_mean_intrinsic_return)

        if self.num_return_samples >= 2:
            self.running_var_intrinsic_return = ((self.num_return_samples - 2)/(self.num_return_samples - 1)) * self.running_var_intrinsic_return
            self.running_var_intrinsic_return += (1/self.num_return_samples) * ((np.sum(intrinsic_rews) - self.running_mean_intrinsic_return)**2)
            self.running_std_intrinsic_return = np.sqrt(self.running_var_intrinsic_return)

            intrinsic_rews /= (self.running_std_intrinsic_return + 1e-8)

        # Compute the advantages.
        with torch.no_grad():
            vals = self._value_function(torch_ify(obs)).cpu().numpy()
            nxt_vals = self._value_function(torch_ify(nxts)).cpu().numpy()
        deltas = rews + self._discount * nxt_vals * (1 - dones) - vals
        advantages = np.zeros(path_len)
        for idx, delta in enumerate(deltas.flatten()[::-1]):
            if idx == 0:
                advantages[-1] = delta
            else:
                advantages[path_len - 1 - idx] = (delta
                        + self._discount * self._tdlambda
                        * advantages[path_len - idx])
        targets = advantages + vals.flatten()
        
        # Compute the intrinsic advantages.
        with torch.no_grad():
            intrinsic_vals = self._intrinsic_value_function(
                torch_ify(obs)).cpu().numpy()
            nxt_intrinsic_vals = self._intrinsic_value_function(
                torch_ify(nxts)).cpu().numpy()
        intrinsic_deltas = intrinsic_rews + self._intrinsic_discount * \
                nxt_intrinsic_vals * (1 - (dones * self.use_dones_for_rnd_critic)) - intrinsic_vals
        intrinsic_advantages = np.zeros(path_len)
        for idx, intrinsic_delta in enumerate(intrinsic_deltas.flatten()[::-1]):
            if idx == 0:
                intrinsic_advantages[-1] = intrinsic_delta
            else:
                intrinsic_advantages[path_len - 1 - idx] = (intrinsic_delta
                        + self._intrinsic_discount * self._tdlambda
                        * intrinsic_advantages[path_len - idx])
        intrinsic_targets = intrinsic_advantages + intrinsic_vals.flatten()
        
        for o, a, r, ir, irr, n, d, ad, iad, pi, t , it in zip(obs, acts, rews, 
                                            intrinsic_rews, intrinsic_rews_raw, nxts, dones, 
                                            advantages, intrinsic_advantages, 
                                            pis, targets, intrinsic_targets):
            self._observations[self._top] = o
            self._actions[self._top] = a
            self._rewards[self._top] = r
            self._intrinsic_rewards[self._top] = ir
            self._intrinsic_rewards_raw[self._top] = irr
            self._terminals[self._top] = d
            self._next_obs[self._top] = n
            self._advantages[self._top] = ad
            self._intrinsic_advantages[self._top] = iad
            self._targets[self._top] = t
            self._intrinsic_targets[self._top] = it
            self._logpis[self._top] = pi

            self._advance()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
