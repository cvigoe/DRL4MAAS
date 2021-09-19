"""
Implementation of Mirror Descent Policy Optimization.
https://arxiv.org/pdf/1707.06347.pdf

Author: Ian Char
Date: April 10, 2021
"""
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

MDPOLosses = namedtuple(
    'MDPOLosses',
    'policy_loss val_loss',
)

class MDPOTrainer(TorchTrainer, LossFunction, metaclass=abc.ABCMeta):
    def __init__(
        self,
        env,
        policy,
        target_policy,
        val,
        epoch_iterations,

        discount=0.99,
        policy_lr=3e-4,
        val_lr=1e-3,
        optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.val = val
        self.discount=discount
        self.val_criterion = nn.MSELoss()
        self.epoch_iterations = epoch_iterations
        self.tk = 0
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.val_optimizer = optimizer_class(
            self.val.parameters(),
            lr=val_lr,
        )
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        ptu.soft_update_from_to(
            self.policy, self.target_policy, 0
        )

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
    ) -> Tuple[MDPOLosses, LossStatistics]:
        obs = batch['observations']
        advantages = batch['advantages']
        targets = batch['targets']
        oldpis = batch['logpis']
        rewards = batch['rewards']
        terminals = batch['terminals']
        actions = batch['actions']
        next_obs = batch['next_observations']
        # Normalize the advanatages.
        advantages = ((advantages - advantages.mean())
                      / (advantages.std() + 1e-8))
        # Compute Policy loss.
        dist = self.policy(obs)
        log_pi = dist.log_prob(actions).unsqueeze(-1)
        weighted_adv = (log_pi - oldpis).exp() * advantages
        # Compute the KL Penalty.
        imaginary_acts, im_currpi = dist.rsample_and_logprob()
        olddist = self.target_policy(obs)
        with torch.no_grad():
            im_oldpi = olddist.log_prob(imaginary_acts).unsqueeze(-1)
        kl_div = im_currpi - im_oldpi
        # Combine to get policy loss.
        policy_loss = torch.mean((-weighted_adv
                       + min(self.tk / self.epoch_iterations, 1) * kl_div))
        # Compute the value loss.
        val_ests = self.val(obs)
        val_loss = self.val_criterion(val_ests, targets)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
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

        loss = MDPOLosses(
            policy_loss=policy_loss,
            val_loss=val_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        # Update the target policy completely.
        ptu.soft_update_from_to(
            self.policy, self.target_policy, 0
        )
        self.tk = 0

    @property
    def networks(self):
        return [
            self.policy,
            self.val,
        ]

    @property
    def optimizers(self):
        return [
            self.val_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            target_policy=self.target_policy,
            val=self.val,
        )
