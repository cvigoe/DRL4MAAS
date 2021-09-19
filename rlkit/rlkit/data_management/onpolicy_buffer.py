"""
Buffer for storing on-policy data.

Author: Ian Char
Date: 4/7/2021
"""
import abc


class OnPolicyBuffer(object, metaclass=abc.ABCMeta):
    """ Class for storing on policy data."""

    @abc.abstractmethod
    def add_path(self, path):
        """
        Add a path to the replay buffer.
        """
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    @abc.abstractmethod
    def clear_buffer(self):
        """Clear the replay buffers."""
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)
