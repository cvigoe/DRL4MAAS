"""
Environments for testing PID type tasks of hitting a target.
"""
import numpy as np
import gym
from simple_pid import PID

from rlkit.envs.env_model import EnvModel


def brownian_gravity(g, f, t):
    return np.clip(g + np.random.normal(0, f * 0.05), 0.35, 0.65)


def static_gravity(g, f, t):
    return g


class SimplePidEnv(gym.Env):
    """
    This is a simple environment where there is a ball with mass that is
    falling and an upward force can be applied.
    """

    def __init__(self, target=5, m=0.75, dt=0.1, g=0.5, horizon=100,
                 mass_updator = lambda m, f, t: m,
                 gravity_updator = lambda g, f, t: g,
                 pid_state=False,
                 pd_obs=True,
                 params_in_state=False,
                 num_hist_in_state=0):
        self.pd_obs = pd_obs
        self.pid_state = pid_state
        self.params_in_state = params_in_state
        self.num_hist_in_state = num_hist_in_state
        obs_dim = (2 * (not pid_state)
                + 4 * (pid_state)
                + 2 * params_in_state
                + num_hist_in_state)
        # Make bounds -1 and 1 for convenience, but this is not true.
        self.observation_space = gym.spaces.Box(
                low=-1 * np.ones(obs_dim),
                high=np.ones(obs_dim),
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
                low=-1, high=1, dtype=np.float32, shape=(1,))
        self.horizon = horizon
        self.target = target
        self.m = m
        self.dt = dt
        self.g = g
        self.mass_updator = mass_updator
        self.gravity_updator = gravity_updator
        self.t = 0
        self.history = []
        self.reset()

    def reset(self, state=None, g=None, m=None,
              mass_updator=None,
              gravity_updator=None):
        self.history = []
        if state is None:
            self.state = (np.array([5, 0])
                          + np.random.uniform([-2.5, -1], [2.5, 1]))
        else:
            self.state = state
        self.t = 0
        self.pid = PID(1, 1, 1, setpoint=self.target)
        self.pid(self.state[0], dt=self.dt)
        if g is not None:
            self.g = g
        if m is not None:
            self.m = m
        if mass_updator is not None:
            self.mass_updator = mass_updator
        if gravity_updator is not None:
            self.gravity_updator = gravity_updator
        for _ in range(self.num_hist_in_state):
            self.step(0)
        self.t = 0
        return self._get_obs()

    def reset_with_target(self, target):
        self.target = target
        return self.reset()

    def step(self, action):
        action = (action + 1) / 2
        self.m = self.mass_updator(self.m, action, self.t)
        self.g = self.gravity_updator(self.g, action, self.t)
        acc = (action - self.g) / self.m
        vel = self.state[1] + acc * self.dt
        posn = np.clip(self.state[0] + vel * self.dt, 0, 10)
        self.state = np.array([posn, vel]).flatten()
        self.pid(self.state[0], dt=self.dt)
        p, i, d = self.pid.components
        self.t += 1
        self.history.append(float(self.state[0]))
        return self._get_obs(), -1 * np.abs(p), False, {}

    def _get_obs(self):
        obs = np.array(self.state).flatten()
        if self.pid_state:
            obs = np.append(obs[:1], np.array(self.pid.components))
        elif self.pd_obs:
            obs[0] = self.target - obs[0]
            obs[1] *= -1
        if self.params_in_state:
            obs = np.append(obs, np.array([float(self.m), float(self.g)]))
        if self.num_hist_in_state > 0:
            for idx in range(1, self.num_hist_in_state + 1):
                if idx > len(self.history):
                    obs = np.append(obs, np.array([0]))
                else:
                    obs = np.append(obs, np.array([self.state[-idx]]))
        return obs


class SimplePidEnvPriorWrapper:

    def __init__(
        self,
        env,
        mass_prior = lambda : 0.75 + np.random.uniform(-0.5, 0.5),
        gravity_prior = lambda: 0.5 + np.random.uniform(-0.15, 0.15),
    ):
        self.env = env
        self.mass_prior = mass_prior
        self.gravity_prior = gravity_prior

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.env.m = self.mass_prior()
        self.env.g = self.gravity_prior()
        return state

    @property
    def state(self):
        return self.env.state

    @state.setter
    def state(self, value):
        self.env.state = value

    @property
    def m(self):
        return self.env.m

    @m.setter
    def m(self, value):
        self.env.m = value

    @property
    def g(self):
        return self.env.g

    @g.setter
    def g(self, value):
        self.env.g = value

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


class SimplePidModelEnv(EnvModel):
    """
    This is a simple environment where there is a ball with mass that is
    falling and an upward force can be applied. This eenvironment is vectorized
    so it has less options available.
    """

    def __init__(self, target=5, dt=0.1, horizon=100,
            mass_prior = lambda shape: 0.75 + np.random.uniform(-0.5, 0.5, shape),
            gravity_prior = lambda shape: 0.5 + np.random.uniform(-0.15, 0.15, shape),
            pd_obs=True,
    ):
        self.pd_obs = pd_obs
        obs_dim = 2
        # Make bounds -1 and 1 for convenience, but this is not true.
        self.observation_space = gym.spaces.Box(
                low=-1 * np.ones(obs_dim),
                high=np.ones(obs_dim),
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
                low=-1, high=1, dtype=np.float32, shape=(1,))
        self.target = target
        self.dt = dt
        self.mass_prior = mass_prior
        self.gravity_prior = gravity_prior

    def multi_step(self, state, actions, masses, gravities):
        actions = (actions + 1) / 2
        accs = (actions - gravities) / masses
        vels = state[:, 1] + accs * self.dt
        posns = np.clip(state[:, 0] + vels * self.dt, 0, 10)
        return {
            'state': np.hstack([posns.reshape(-1, 1), vels.reshape(-1, 1)]),
            'obs': np.hstack([self.target - posns.reshape(-1, 1),
                              -1 * vels.reshape(-1, 1)]),
            'rewards': -1 * np.abs(posns - self.target),
        }


    def unroll(self, start_states, policy, horizon,
               replay_buffer=None, actions=None):
        """Unroll for multiple trajectories at once.
        Args:
            start_states: The start states to unroll at as ndarray
                w shape (num_starts, obs_dim).
            policy: Policy to take actions.
            horizon: How long to rollout for.
            replay_buffer: Replay buffer to add to.
            actions: The actions to use to unroll.
        """
        should_call_policy = actions is None
        # Draw the masses and gravities.
        masses = self.mass_prior(start_states.shape[0])
        gravities = self.gravity_prior(start_states.shape[0])
        # Init the datastructures.
        states = np.zeros((horizon + 1, start_states.shape[0], 2))
        obs = np.zeros((horizon + 1, start_states.shape[0], 2))
        actions = np.zeros((horizon, start_states.shape[0]))
        states[0] = start_states
        obs[0] = start_states
        if self.pd_obs:
            obs[0, :, 0] = self.target - obs[0, :, 0]
            obs[0, :, 1] *= -1
        rewards = np.zeros((horizon, start_states.shape[0]))
        terminals = np.full((horizon, start_states.shape[0]), False)
        logpis = np.zeros((horizon, start_states.shape[0]))
        for hidx in range(horizon):
            # Get actions for each of the states.
            if should_call_policy:
                net_in = obs[hidx] if self.pd_obs else state[hidx]
                acts, probs = policy.get_actions(net_in)
                acts = acts.flatten()
                actions[hidx] = acts
                logpis[hidx] = probs
            else:
                acts = actions[hidx]
            # Roll all states forward.
            nxt_info = self.multi_step(states[hidx], acts, masses, gravities)
            states[hidx+1] = nxt_info['state']
            obs[hidx+1] = nxt_info['obs']
            rewards[hidx] = nxt_info['rewards']
        # Add to replay buffer
        if replay_buffer is not None:
            for pathnum in range(obs.shape[1]):
                replay_buffer.add_path(dict(
                    observations=obs[:-1, pathnum],
                    next_observations=obs[1:, pathnum],
                    actions=actions[:, pathnum],
                    rewards=rewards[:, pathnum].reshape(-1, 1),
                    terminals=terminals[:, pathnum].reshape(-1, 1),
                    env_infos=[{} for _ in range(len(actions))],
                    agent_infos=[{'logpi': lp} for lp in logpis[:, pathnum]],
                ))
        # Log stats.
        return obs, actions, rewards, {}

