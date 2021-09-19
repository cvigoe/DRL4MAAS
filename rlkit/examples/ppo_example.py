"""
Example for using PPO.

Author: Ian Char
Date: 4/9/2021
"""
import argparse
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.advantage_buffer import AdvantageReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.policy_gradient import PPOTrainer
from rlkit.torch.networks import Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(env_name, variant):
    eval_env = NormalizedBoxEnv(gym.make(env_name))
    expl_env = NormalizedBoxEnv(gym.make(env_name))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    valf = Mlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        **variant['policy_kwargs'],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = AdvantageReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        valf,
        discount=variant['trainer_kwargs']['discount'],
        **variant['target_kwargs']
    )
    trainer = PPOTrainer(
        env=eval_env,
        policy=policy,
        val=valf,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Hopper-v2')
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--fixed_std', type=float)
    parser.add_argument('--pudb', action='store_true')
    args = parser.parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    variant = dict(
        algorithm="PPO",
        version="normal",
        layer_size=args.layer_size,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=2048,
            min_num_steps_before_training=0,
            num_train_loops_per_epoch=10,
            max_path_length=args.max_path_length,
            batch_size=256,
            clear_buffer_every_train_loop=True,
        ),
        trainer_kwargs=dict(
            epsilon=0.2,
            discount=args.discount,
            policy_lr=3E-4,
            val_lr=3E-4,
        ),
        target_kwargs=dict(
            tdlambda=0.95,
            target_lookahead=15,
        ),
        policy_kwargs=dict(
            std=args.fixed_std,
        ),
    )
    setup_logger('ppo-%s-baseline' % args.env, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(args.env, variant)
