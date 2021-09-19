"""
Class for environments that are models.
"""
import abc


class EnvModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def unroll(self, start_states, policy, horizon, actions=None):
        """Unroll for multiple trajectories at once.

        Args:
            start_states: The start states to unroll at as ndarray
                w shape (num_starts, obs_dim).
            policy: Policy to take actions.
            horizon: How long to rollout for.
            actions: The actions to use to unroll.

        Returns:
            * obs ndarray of (horizon + 1, num_starts, obs_dim)
            * actions ndarray of (horizon, num_starts, act_dim)
            * rewards ndarray of (horizon, num_starts)
            * terminals ndarray of (horizon, num_starts)
            * env_info mapping from str -> ndarray
            * actor_info mapping str -> ndarray
        """

    def get_diagnostics(self, epoch):
        return {}

    def end_epoch(self, epoch):
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass
