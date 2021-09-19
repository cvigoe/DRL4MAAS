from collections import OrderedDict

import torch.optim as optim
import torch.nn as nn

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.core import np_ify
from rlkit.torch.PETS.model import gaussian_log_loss


class DynModelTrainer(TorchTrainer):
    def __init__(
            self,
            model,
            lr=1e-3,
            optimizer_class=optim.Adam,
            validation_set=None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mean_criterion = nn.MSELoss()  # just for information
        self.model_criterion = gaussian_log_loss
        self.model_optimizer = optimizer_class(
                self.model.parameters(),
                lr=lr,
        )
        self._validation_set = validation_set

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # in order to bootstrap the models, we need to train one network only per batch
        net_idx = self._n_train_steps_total % len(self.model._nets)
        mean, logvar, predcted_rewards = self.model.forward(obs, actions, network_idx=net_idx, return_net_outputs=True)
        # TODO: possibly need to include weight decay
        mean_mse = self.mean_criterion(mean, next_obs)

        model_loss = self.model_criterion(mean, logvar, next_obs)
        bound_loss = self.model.bound_loss()
        # TODO: Right now I just have reward_loss 0 but if we want
        # to do this in the future, there are things to be done here.
        reward_loss = 0
        loss = model_loss + bound_loss + reward_loss
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        self.model.trained_at_all = True

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['Model Loss'] = np_ify(model_loss)
            self.eval_statistics['Bound Loss'] = np_ify(bound_loss)
            self.eval_statistics['Reward Loss'] = np_ify(reward_loss)
            self.eval_statistics['Model MSE'] = np_ify(mean_mse)
            self.eval_statistics['Loss'] = np_ify(loss)
            # TODO: Add logging of validation loss.
        self._n_train_steps_total += 1

    @property
    def networks(self):
        return list(self.model._nets)

    def get_snapshot(self):
        return {'net%d' % net_idx: net
                for net_idx, net in enumerate(self.networks)}

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
