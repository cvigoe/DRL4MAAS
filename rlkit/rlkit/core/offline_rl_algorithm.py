"""
RL Algorithm for learning from offline data.

Author: Ian Char
Date: 9/8/2020
"""
import abc

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.data_management.offline_data_store import OfflineDataStore
from rlkit.samplers.data_collector import DataCollector


class OfflineRLAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            offline_data,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_train_loops_per_epoch,
            max_path_length,
            evaluation_env=None,
            evaluation_data_collector: DataCollector=None,
    ):
        self.trainer = trainer
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.offline_data = offline_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.max_path_length = max_path_length
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """Training of the policy implemented by child class."""
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.eval_data_collector is not None:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                gt.stamp('evaluation sampling')

            self.training_mode(True)
            gt.stamp('training', unique=False)
            for _ in range(self.num_train_loops_per_epoch):
                train_data = self.offline_data.random_batch(self.batch_size)
                self.trainer.train(train_data)
            self.training_mode(False)

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        if self.eval_data_collector is not None:
            self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        if self.eval_data_collector is not None:
            for k, v in self.eval_data_collector.get_snapshot().items():
                snapshot['evaluation/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        """
        Evaluation
        """
        if self.eval_data_collector is not None:
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if (self.eval_env is not None
                and hasattr(self.eval_env, 'get_diagnostics')):
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
