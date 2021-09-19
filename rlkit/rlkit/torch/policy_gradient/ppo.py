"""
Policy Gradient algorithms.

Author: Ian Char
Date: 4/7/2021
"""
import abc
from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt

import pudb

PPOLosses = namedtuple(
    'PPOLosses',
    'policy_loss val_loss intrinsic_val_loss rnd_loss',
)

class PPOTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        env,
        policy,
        val,
        intrinsic_val,
        rnd,

        use_rnd=False,
        rnd_coef=1,
        predictor_update_proportion=0.25,

        epsilon=0.2,
        discount=0.99,
        intrinsic_discount=0.9999,

        policy_lr=3e-4,
        val_lr=1e-3,
        intrinsic_val_lr=1e-3,
        optimizer_class=optim.Adam,

        entropy_bonus=0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.val = val
        self.rnd = rnd
        self.use_rnd = use_rnd
        self.rnd_coef = rnd_coef
        self.intrinsic_val = intrinsic_val
        self.epsilon = epsilon
        self.entropy_bonus = entropy_bonus
        self.discount=discount
        self.val_criterion = nn.MSELoss()
        self.intrinsic_val_criterion = nn.MSELoss()
        self.predictor_update_proportion = predictor_update_proportion
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.val_optimizer = optimizer_class(
            self.val.parameters(),
            lr=val_lr,
        )
        self.intrinsic_val_optimizer = optimizer_class(
            self.intrinsic_val.parameters(),
            lr=intrinsic_val_lr,
        )
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.val_optimizer.zero_grad()
        losses.val_loss.backward()
        self.val_optimizer.step()

        self.intrinsic_val_optimizer.zero_grad()
        losses.intrinsic_val_loss.backward()
        self.intrinsic_val_optimizer.step()        

        if self.use_rnd:
            self.rnd.optimiser.zero_grad()
            losses.rnd_loss.backward()
            self.rnd.optimiser.step()

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('ppo training', unique=False)

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[PPOLosses, LossStatistics]:
        obs = batch['observations']
        advantages = batch['advantages']
        intrinsic_advantages = batch['intrinsic_advantages']
        targets = batch['targets']
        intrinsic_targets = batch['intrinsic_targets']
        oldpis = batch['logpis']
        rewards = batch['rewards']
        intrinsic_rewards = batch['intrinsic_rewards']
        intrinsic_rewards_raw = batch['intrinsic_rewards_raw']
        terminals = batch['terminals']
        actions = batch['actions']
        next_obs = batch['next_observations']
        
        # Normalize the advanatages.
        # NOTE: Doesn't look like OG paper or PyTorch implementation uses int or ext advantage normalisations

        advantages = ((advantages - advantages.mean())
                      / (advantages.std() + 1e-8))

        # NOTE: as per https://arxiv.org/pdf/2006.05990.pdf, we should
        # perform continual dimension-wise observation whitening. Note
        # also that for now am using the rnd obs normaliser as it is 
        # convenient. Note that obs are already whitened in buffer.
        # obs = self.rnd.observation_normaliser(obs)

        # Compute Policy loss.
        dist = self.policy(obs)
        log_pi = dist.log_prob(actions).unsqueeze(-1)
        ratio = (log_pi - oldpis).exp()
        policy_loss = -1 * torch.mean(torch.min(
            ratio * (advantages + intrinsic_advantages*self.use_rnd*self.rnd_coef),
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * (advantages + intrinsic_advantages*self.use_rnd*self.rnd_coef),
        ) + self.entropy_bonus * log_pi)
        kl_div = (log_pi.detach() - oldpis)
        
        # Compute the value loss.
        val_ests = self.val(obs)
        val_loss = self.val_criterion(val_ests, targets)

        # Compute the intrinsic value loss.
        intrinsic_val_ests = self.intrinsic_val(obs)
        intrinsic_val_loss = self.intrinsic_val_criterion(
            intrinsic_val_ests, intrinsic_targets)

        # Compute the RND loss
        if self.use_rnd:
            mask = torch.rand(len(obs))
            mask = (mask < self.predictor_update_proportion).type(
                torch.FloatTensor)

            rnd_loss = self.val_criterion(
                self.rnd.prediction_network(obs)[mask.byte()],
                self.rnd.random_network(obs)[mask.byte()]
                )
        else:
            rnd_loss = None

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Extrinsic Rewards Mean'] = np.mean(ptu.get_numpy(rewards))
            eval_statistics['Extrinsic Rewards Max'] = np.max(ptu.get_numpy(rewards))
            eval_statistics['Intrinsic Rewards Mean'] = np.mean(ptu.get_numpy(intrinsic_rewards))
            eval_statistics['Intrinsic Rewards Max'] = np.max(ptu.get_numpy(intrinsic_rewards))
            eval_statistics['Intrinsic Rewards Raw Mean'] = np.mean(ptu.get_numpy(intrinsic_rewards_raw))
            eval_statistics['Intrinsic Rewards Raw Max'] = np.max(ptu.get_numpy(intrinsic_rewards_raw))            
            eval_statistics['Intrinsic Advantages Mean'] = np.mean(ptu.get_numpy(intrinsic_advantages))
            eval_statistics['Intrinsic Advantages Max'] = np.max(ptu.get_numpy(intrinsic_advantages))
            eval_statistics['Intrinsic Targets Mean'] = np.mean(ptu.get_numpy(intrinsic_targets))
            eval_statistics['Intrinsic Targets Max'] = np.max(ptu.get_numpy(intrinsic_targets))

            eval_statistics['Extrinsic Advantages Mean'] = np.mean(ptu.get_numpy(advantages))
            eval_statistics['Extrinsic Advantages Max'] = np.max(ptu.get_numpy(advantages))

            if rnd_loss:
                eval_statistics['RND Loss'] = np.mean(ptu.get_numpy(rnd_loss))
            eval_statistics['Intrinsic Value Loss'] = np.mean(ptu.get_numpy(intrinsic_val_loss))
            eval_statistics['Value Loss'] = np.mean(ptu.get_numpy(val_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Value Estimates',
                ptu.get_numpy(val_ests),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'KL Div',
                ptu.get_numpy(kl_div),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)

        loss = PPOLosses(
            policy_loss=policy_loss,
            val_loss=val_loss,
            intrinsic_val_loss=intrinsic_val_loss,
            rnd_loss=rnd_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.val,
            self.intrinsic_val,
        ]

    @property
    def optimizers(self):
        return [
            self.val_optimizer,
            self.intrinsic_val_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            val=self.val,
            intrinsic_val=self.intrinsic_val,
        )
