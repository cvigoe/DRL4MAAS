import gym

from rlkit.envs.custom.backward_env import BackwardHalfCheetah,\
        BackwardHopper, BackwardWalker
from rlkit.envs.custom.fast_pendulum_env import FastPendulum

gym.envs.register(
    id='BackwardHalfCheetah-v2',
    entry_point='rlkit.envs.custom.backward_env:BackwardHalfCheetah',
)

gym.envs.register(
    id='BackwardHopper-v2',
    entry_point='rlkit.envs.custom.backward_env:BackwardHopper',
)

gym.envs.register(
    id='BackwardWalker-v2',
    entry_point='rlkit.envs.custom.backward_env:BackwardWalker',
)

gym.envs.register(
    id='FastPendulum-v0',
    entry_point='rlkit.envs.custom.fast_pendulum_env:FastPendulum',
)

CUSTOM_ENVS = ['BackwardHalfCheetah', 'BackwardHopper', 'BackwardWalker',
               'FastPendulum']
