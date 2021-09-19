"""
Halfcheetah environment with rewards reversed so agent is rewarded for
going backwards as fast as possible.
"""
import gym


class BackwardHalfCheetah(gym.Env):

    def __init__(self):
        self._env = gym.make('HalfCheetah-v2')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        nxt, rew, done, info = self._env.step(action)
        return nxt, -rew, done, info

class BackwardHopper(gym.Env):

    def __init__(self):
        self._env = gym.make('Hopper-v2')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        nxt, rew, done, info = self._env.step(action)
        return nxt, -rew, done, info

class BackwardWalker(gym.Env):

    def __init__(self):
        self._env = gym.make('Walker2d-v2')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        nxt, rew, done, info = self._env.step(action)
        return nxt, -rew, done, info
