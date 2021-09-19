"""
Boilerplate code for doing offline experiment.

Author: Ian Char
Date: 9/10/2020
"""
import argparse
import os
import pickle as pkl

import d4rl
import h5py
import gym
import numpy as np

from rlkit.core import logger
from rlkit.data_management.offline_data_store import OfflineDataStore
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.cql.cql import CQLTrainer
from rlkit.torch.networks import ConcatMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchOfflineRLAlgorithm


def assemble_rl_trainer(env, variant):
    """Assemble trainer for doing RL."""
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        env,
        eval_policy,
    )
    if variant['algorithm'] == 'AWAC':
        trainer_class = AWACTrainer
    elif variant['algorithm'] == 'SAC':
        trainer_class = SACTrainer
    elif variant['algorithm'] == 'CQL':
        trainer_class = CQLTrainer
        variant['trainer_kwargs']['min_q_weight'] = 5.0
    else:
        raise ValueError('Unknown algorithm %s' % variant['algorithm'])
    trainer = trainer_class(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    return trainer, eval_path_collector


def run_experiment(variant):
    """Run the experiment."""
    save_path = variant['run_id']
    # Set device and get data.
    use_gpu = variant['cuda_device'] != ''
    ptu.set_gpu_mode(use_gpu, gpu_id=variant['cuda_device'])
    env = gym.make(variant['env'])
    dataset = d4rl.qlearning_dataset(env.env)
    if variant['data_path'] is not None and variant['amt_to_add'] > 0:
        with h5py.File(variant['data_path'], 'r') as hdata:
            for k, v in hdata.items():
                if k in dataset:
                    if len(dataset[k].shape) == 2:
                        dataset[k] = np.vstack(
                                [dataset[k], v[:variant['amt_to_add']]])
                    else:
                        dataset[k] = np.append(dataset[k],
                                v[:variant['amt_to_add']].flatten())
    if variant['save_every'] > 0:
        setup_logger(
                save_path,
                variant=variant,
                snapshot_mode='gap_and_last',
                snapshot_gap=variant['save_every'],
        )
    else:
        setup_logger(
                save_path,
                variant=variant,
        )
    trainer, path_collector = assemble_rl_trainer(env, variant)
    offline_data = OfflineDataStore(data=dataset)
    algorithm = TorchOfflineRLAlgorithm(
        trainer=trainer,
        evaluation_env=env,
        evaluation_data_collector=path_collector,
        offline_data=offline_data,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def parse_run_options():
    """Parse run options."""
    parser = argparse.ArgumentParser('Offline Experiment Options.')
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default=None)
    parser.add_argument('--data_path')
    # Percent of the data from data_path to add.
    parser.add_argument('--amt_to_add', type=int, default=0)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        qf_architecture=[256, 256],
        policy_architecture=[256, 256],
        penalize=False,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=10000,
            num_train_loops_per_epoch=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, 256]
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256],
        ),
    )
    variant.update(vars(parse_run_options()))
    if variant['pudb']:
        import pudb; pudb.set_trace()
    run_experiment(variant)
