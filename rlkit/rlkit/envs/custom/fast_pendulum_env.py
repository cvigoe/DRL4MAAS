"""
An environment that rewards for getting the policy to go as fast as possible.
"""
import gym
import numpy as np


class FastPendulum(gym.Env):

    def __init__(self):
        self._env = gym.make('Pendulum-v0')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        nxt, rew, done, info = self._env.step(action)
        nrew = np.abs(nxt[-1])
        return nxt, nrew, done, info
