import argparse
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.advantage_buffer import AdvantageReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.policy_gradient import PPOTrainer
from rlkit.torch.networks import Mlp, MlpRND
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import gym_activesearchrlpoisson

import mlflow
from mlflow.tracking import MlflowClient
import sys
import collections
import pudb

import torch
import numpy as np

from variant import *


class RND():
    def __init__(self, obs_dim, rnd_output_size, M, LR, use_normaliser):
        self.observation_normaliser = ObservationNormaliser(obs_dim=obs_dim)
        self.use_normaliser = use_normaliser
        self.random_network = MlpRND(
            input_size=obs_dim,
            output_size=rnd_output_size,
            hidden_sizes=[M, M],
            frozen=True,
        )
        self.prediction_network = MlpRND(
            input_size=obs_dim,
            output_size=rnd_output_size,
            hidden_sizes=[M, M],
            frozen=False,
        )
        self.optimiser = torch.optim.Adam(self.prediction_network.parameters(),lr=LR)
        self.loss = torch.nn.MSELoss()

    def intrinsic_reward(self, observation):
        observation = self.observation_normaliser(observation)
        observation = torch.Tensor(observation)
        return self.loss(
            self.prediction_network(observation) ,
            self.random_network(observation)  
            )

class ObservationNormaliser():
    def __init__(self, obs_dim, mean=0, std=0.1, num_observations=0):
        self.obs_dim = obs_dim        
        self.mean = torch.Tensor([mean]*obs_dim)
        self.std = torch.Tensor([std]*obs_dim)
        self.var = torch.Tensor([.001]*obs_dim)
        self.num_observations = num_observations

    def __call__(self, observations):
        if len(observations.shape) == 1:   
            return np.clip( (observations - self.mean.numpy()) / (self.std.numpy() + 1e-7) , -5,5 ) 
        for i in range(observations.shape[0]):
            observation = observations[i,:]
            delta = observation - self.mean
            self.num_observations += 1
            self.mean += (1/self.num_observations) * delta  # See page 53 of Sutton RL 2nd Edition
            if self.num_observations >= 2:                  # https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
                self.var = ((self.num_observations - 2)/(self.num_observations - 1)) * self.var
                self.var += (1/self.num_observations) * (delta**2)
            self.std = np.sqrt(self.var)
        return np.clip( (observations - self.mean) / (self.std + 1e-7) , -5,5 )



def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def experiment(variant, env_variant):
    eval_env = gym.make(env_variant['env_str']) # May need to add NormalizedBox()
    expl_env = gym.make(env_variant['env_str']) # May need to add NormalizedBox()
    if env_variant['env_str'] == 'activesearchrlpoissonmlemap-v0' \
        or env_variant['env_str'] == 'agnosticmaas-v0':
        expl_env.inialise_environment(**env_variant)
        eval_env.inialise_environment(**env_variant)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M_actor = variant['actor_width']
    M_critic = variant['critic_width']
    valf = Mlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M_critic, M_critic],
    )
    intrinsic_valf = Mlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M_critic, M_critic],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M_actor, M_actor],
        **variant['policy_kwargs'],
    )
    rnd = RND(obs_dim, variant['rnd_kwargs']['rnd_output_size'], 
        variant['rnd_kwargs']['rnd_latent_size'], 
        variant['rnd_kwargs']['rnd_lr'], 
        variant['rnd_kwargs']['use_normaliser'])
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        rnd,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        rnd,
    )
    replay_buffer = AdvantageReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        valf,
        intrinsic_valf,
        discount=variant['trainer_kwargs']['discount'],
        intrinsic_discount=variant['trainer_kwargs']['intrinsic_discount'],
        **variant['target_kwargs']
    )
    trainer = PPOTrainer(
        env=eval_env,
        policy=policy,
        val=valf,
        intrinsic_val=intrinsic_valf,
        rnd=rnd,
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
    experiment_name = sys.argv[1]
    run_name = sys.argv[2]
    note = sys.argv[3]

    setup_logger(experiment_name, variant=variant)
    if variant['gpu']:
        ptu.set_gpu_mode(True)
    mlflow.set_tracking_uri(variant['mlflow_uri'])
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()  
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_dict(variant))
        mlflow.log_params(flatten_dict(env_variant))
        client.set_tag(run.info.run_id, "mlflow.note.content", note)
        experiment(variant, env_variant)
