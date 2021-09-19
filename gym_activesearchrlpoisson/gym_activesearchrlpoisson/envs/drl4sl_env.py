"""
DRL4SL Gym Env

Author: Conor Igoe
Date: 04/27/2021
"""
import pudb
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch

class DRL4SL(gym.Env):
    """Initialises the DRL4SL toy problem

    Attributes:
        c1, c2 ,c3: 
          scalar params of the objective function
        init_x1, init_x2: 
          scalar initial conditions for x1 and x2
        num_timesteps: 
          intieger specifying when a particular episode terminates
        state_dim: 
          integer specifying dimension of state
    """    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, c1: float=1, c2: float=2, c3: float=2, init_x1: float=2.0, 
                 init_x2: float=2.0, num_timesteps: int=50, state_dim: int=4):
        super(DRL4SL, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c1_init = c1
        self.c2_init = c2
        self.c3_init = c3
        self.init_x1 = init_x1
        self.init_x2 = init_x2
        self.num_timesteps = num_timesteps
        self.state_dim = state_dim

        self.action_space = spaces.Box( np.array([-1*np.inf]), np.array([np.inf]) )
        self.observation_space = spaces.Box( np.array([-1*np.inf]*self.state_dim), np.array([np.inf]*self.state_dim) )

    def inialise_environment(self, env_str, param_std):
        self.param_std = param_std

    def reset(self):
        """Resets the gym environment, initialises x1 and x2,
        and redraws params for the objective (not implemetned yet)

        Returns:
          state: 
            numpy array of scalar values of length self.state_dim
        """
        self.t = 0
        eps1 = np.random.randn()*self.param_std*0
        eps2 = np.random.randn()*self.param_std*0
        self.x1 = torch.tensor([self.init_x1 + eps1], requires_grad=True)
        self.x2 = torch.tensor([self.init_x2 + eps2], requires_grad=True)

        self.c1 = self.c1_init + np.random.randn()*self.param_std
        self.c2 = self.c2_init + np.random.randn()*self.param_std
        self.c3 = self.c3_init + np.random.randn()*self.param_std

        y = self.objective_function(self.x1,self.x2)
        y.backward()

        self.x1.grad.data.zero_()
        self.x2.grad.data.zero_()        

        state = [self.x1.detach().item(), self.x2.detach().item(), self.x1.grad.detach().item(), self.x2.grad.detach().item()]  # to start with just get current t-step info, but then compare against using prev k t-steps

        return np.array(state)

    def step(self, action, verbose=True):
        """Takes a step in the DRL4SL toy problem.

        Args:
            action:
              The learning rate of gradient descent at the current timestep for 
              both x1 and x2. Could make more complex in the future.

        Returns:

          A tuple (state, reward, done, info). The state variable just contains 
          current x1 and x2 and the gradient for now. The reward is the negative 
          objective value. Episode terminates when self.num_timesteps is exceeded.

        """ 

        self.t += 1

        y = self.objective_function(self.x1, self.x2)
        y.backward()

        with torch.no_grad():
            self.x1 -= (1 + action)*self.x1.grad # trying to see if making action be the delta from standard LR is good
            self.x2 -= (1 + action)*self.x2.grad

        self.x1.grad.data.zero_()
        self.x2.grad.data.zero_()

        y_new = self.objective_function(self.x1, self.x2)
        y_new.backward()
        with torch.no_grad():
            state = [self.x1.detach().item(), self.x2.detach().item(), self.x1.grad.detach().item(), self.x2.grad.detach().item()]  # to start with just get current t-step info, but then compare against using prev k t-steps
            self.x1.grad.data.zero_()
            self.x2.grad.data.zero_()
            
            reward = -1*y_new.detach().numpy()[0]
            done = (self.t > self.num_timesteps)

            return np.array(state), np.array(reward), np.array(done), {}

    def objective_function(self, x1, x2):
        numerator = x1
        denominator = torch.log(((x2+self.c1)**self.c2)+self.c3)
        return torch.sin(numerator/denominator)


