"""
Trainer for doing on-policy Value Estimation via TD Lambda.

Author: Ian Char
Date: 4/11/2021
"""
from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import gtimer as gt

ValLosses = namedtuple(
    'ValLosses',
    'val_loss',
)


class ValueTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vf,
            target_vf,
            nsteps,
            lmbda,

            discount=0.99,
            reward_scale=1.0,

            vf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()
        self.vf = vf
        self.target_vf = target_vf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.nsteps = nsteps
        self.lmbda = lmbda
        self._gammas = torch.pow(torch.ones(nsteps) * discount,
                                 torch.arange(nsteps))

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.vf_criterion = torch.nn.MSELoss()

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
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
        self.vf_optimizer.zero_grad()
        losses.val_loss.backward()
        self.vf_optimizer.step()
        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('value training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.vf, self.target_vf, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[ValLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        next_obs = batch['next_observations']
        steps = batch['steps']
        batch_size = len(obs)
        # Get the target values.
        with torch.no_grad():
            val_out = self.target_vf(next_obs.reshape(-1, obs.shape[1]))
            val_out = val_out.reshape(-1, self.nsteps)
        # Put into matrices with 0 indices past information.
        target_vals = ptu.zeros((batch_size, self.nsteps))
        rewmat = ptu.zeros((batch_size, self.nsteps))
        coefs = ptu.zeros((batch_size, self.nsteps))
        for validx, valrow in enumerate(val_out):
            end = int(steps[validx].item())
            target_vals[validx, :end] = valrow[:end]
            rewmat[validx, :end] = rewards[validx, :end]
            if terminals[validx].item():
                target_vals[validx, end - 1] = 0
            if end > 1:
                coefs[validx, :end] =\
                        torch.pow(torch.ones(end) * self.lmbda,
                                  torch.arange(end))
                coefs[validx, :end - 1] *= 1 - self.lmbda
            else:
                coefs[validx, 0] = 1
        # Get empirical returns.
        rets = ptu.zeros(batch_size)
        for st in range(1, self.nsteps):
            rets += (coefs[:, st - 1]
                    * (torch.sum(rewmat[:, :st] * self._gammas[:st], dim=1)
                       + (self._gammas[st] * target_vals[:, st - 1])))
        empirical_ret = torch.sum(rewmat * self._gammas, dim=1)
        val_preds = self.vf(obs)
        val_loss = self.vf_criterion(val_preds, rets.unsqueeze(-1))
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Val Loss'] = np.mean(ptu.get_numpy(val_loss))
            eval_statistics.update(create_stats_ordered_dict(
                'Value Predicts',
                ptu.get_numpy(val_preds),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Value Targets',
                ptu.get_numpy(rets),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Empirical Returns',
                ptu.get_numpy(empirical_ret),
            ))

        loss = ValLosses(val_loss=val_loss,)

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
            self.vf,
            self.target_vf,
        ]

    @property
    def optimizers(self):
        return [
            self.vf_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            vf=self.vf,
            target_vf=self.target_vf,
        )
