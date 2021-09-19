"""
Model based offline RL algorithm.

Author: Ian Char
Date: 11/15/2020
"""
import abc

import numpy as np
import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.offline_data_store import OfflineDataStore
from rlkit.envs.env_model import EnvModel
from rlkit.samplers.data_collector import DataCollector


class MBOfflineRLAlgorithm(metaclass=abc.ABCMeta):
    """Model-based offline algorithm that trains on both stored data and model data."""

    def __init__(
            self,
            trainer,
            model_env: EnvModel,
            evaluation_env,
            model_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            offline_data: OfflineDataStore,
            model_replay_buffer: ReplayBuffer,
            num_epochs,
            offline_batch_size,
            model_batch_size,
            num_model_steps_per_epoch,
            num_eval_steps_per_epoch,
            num_train_loops_per_epoch,
            model_max_path_length,
            eval_max_path_length,
            min_model_steps_before_training,
            pre_model_training_epochs=0,
    ):
        self.trainer = trainer
        self.model_env = model_env
        self.model_data_collector = model_data_collector
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.offline_data = offline_data
        self.model_replay_buffer = model_replay_buffer
        self.num_epochs = num_epochs
        self.offline_batch_size = offline_batch_size
        self.model_batch_size = model_batch_size
        self.num_model_steps_per_epoch = num_model_steps_per_epoch
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.model_max_path_length = model_max_path_length
        self.eval_max_path_length = eval_max_path_length
        self.min_steps_rollouts_before_training =\
                min_steps_rollouts_before_training
        self.pre_model_training_epochs = pre_model_training_epochs
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """Training of the policy implemented by child class."""
        # Train in the normal offline way without any models first.
        for epoch in gt.timed_for(
                range(self._start_epoch, self.pre_model_training_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            self.training_mode(True)
            gt.stamp('training', unique=False)
            for _ in range(self.num_train_loops_per_epoch):
                train_data = self.offline_data.random_batch(
                        self.offline_batch_size + self.model_batch_size)
                self.trainer.train(train_data)
            self.training_mode(False)
            self._end_epoch(epoch)
        # Train with models.
        self.model_data_collector.collect_new_paths(
            self.model_max_path_length,
            self.min_model_steps_before_training,
            discard_incomplete_paths=False,
        )
        for epoch in gt.timed_for(
                range(self.pre_model_training_epochs, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            gt.stamp('model rollout', unique=False)
            self.model_data_collector.collect_new_paths(
                self.model_max_path_length,
                self.num_model_steps_per_epoch,
                discard_incomplete_paths=False,
            )
            # Train over samples.
            self.training_mode(True)
            gt.stamp('training', unique=False)
            for _ in range(self.num_train_loops_per_epoch):
                offline_train_data = self.offline_data.random_batch(
                        self.offline_batch_size)
                model_train_data = self.model_replay_buffer.random_batch(
                        self.model_batch_size)
                train_data = {k: np.vstack([
                    offline_train_data[k],
                    model_train_data[k]]) for k in model_train_data}
                self.trainer.train(train_data)
            self.training_mode(False)

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)
        self.model_env.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        """
        Model
        """
        logger.record_dict(
                self.model_env.get_diagnostics(),
                prefix='model/',
        )
        """
        Model Unroll
        """
        logger.record_dict(
                self.model_replay_buffer.get_diagnostics(),
                prefix='model_replay_buffer/',
        )
        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
