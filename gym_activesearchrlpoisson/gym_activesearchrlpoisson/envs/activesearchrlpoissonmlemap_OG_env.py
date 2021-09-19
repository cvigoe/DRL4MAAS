"""
Using Poisson Point Process representation

Author: Conor Igoe
Date: 04/13/2021
"""
import pudb
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import math
import scipy.stats as ss
from scipy.stats import invgauss
import tqdm
import copy
import networkx as nx
import torch
import scipy as sp

class ActiveSearchRLPoissonMLEMAP(gym.Env):
    """Initialises the Active Search Poisson class with the passed values

    Attributes:
        lam: 
          the homogenous Poisson Point Process rate parameter. Must be > 0.
        sigma2: 
          the variance of the zero mean observation Gaussian noise. Must 
          be >= 0.
        num_agents: 
          number of agents. Must be > 0.
        num_hypotheses: 
          budget of hypotheses. Once this budget has been reached, we must 
          choose which hypotheses to keep as new observsations are made. Must 
          be > 0.
        num_timesteps: 
          specifies when a particular episode terminates. Must be > 0.
        num_EA_iterations: 
          max number of full sweeps through each RV in the conditional.
        EA_tolerance:
          tolerance to break out of EA. Must be > 0.
        cost_iterations: 
          number of MC samples to draw from "posterior" to estimate cost.
        upper_limit_N: 
          upper limit to use when determining expectations from conditional 
          probabilities.
        log_space_resolution: 
          largest delta to represent in logspace when performing EA. Must 
          be > 0.
        MLE_regularizer:
          how much to bias MLE estimation to prior mean
    """    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, lam: float=10, sigma2: float=0.05, num_agents: int=2, 
                 num_hypotheses: int=10, num_timesteps: int=50, 
                 num_EA_iterations: int=10, EA_tolerance: float=0.0001, 
                 cost_iterations: int=20, upper_limit_N: int=10, 
                 log_space_resolution: int=100, MLE_regularizer: float=0.00001):
        super(ActiveSearchRLPoissonMLEMAP, self).__init__()
        self.num_agents = num_agents
        self.num_hypotheses = num_hypotheses
        self.lam = lam
        self.sigma2 = sigma2
        self.num_timesteps = num_timesteps
        self.num_EA_iterations = num_EA_iterations
        self.EA_tolerance = EA_tolerance
        self.cost_iterations = cost_iterations
        self.upper_limit_N = upper_limit_N
        self.log_space_resolution = log_space_resolution
        self.MLE_regularizer = MLE_regularizer

        self.action_space = spaces.Box( np.array([0,0]*self.num_agents), np.array([1,1]*self.num_agents) )
        self.observation_space = spaces.Box( np.array([0]*self.num_hypotheses*2), np.array([np.inf]*self.num_hypotheses + [1]*self.num_hypotheses) )


    def inialise_environment(self, lam: float=10, sigma2: float=0.05, num_agents: int=2,
                 num_hypotheses: int=10, num_timesteps: int=50,
                 num_EA_iterations: int=10, EA_tolerance: float=0.0001,
                 cost_iterations: int=20, upper_limit_N: int=10,
                 log_space_resolution: int=100, env_str: str='activesearchrlpoissonmlemap-v0'):
        self.lam = lam
        self.sigma2 = sigma2
        self.num_agents = num_agents
        self.num_hypotheses = num_hypotheses
        self.num_timesteps = num_timesteps
        self.num_EA_iterations = num_EA_iterations
        self.EA_tolerance = EA_tolerance
        self.cost_iterations = cost_iterations
        self.upper_limit_N = upper_limit_N
        self.log_space_resolution = log_space_resolution

        self.action_space = spaces.Box( np.array([0,0]*self.num_agents), np.array([1,1]*self.num_agents) )
        self.observation_space = spaces.Box( np.array([0]*self.num_hypotheses*2), np.array([np.inf]*self.num_hypotheses + [1]*self.num_hypotheses) )

    def reset(self, verbose=True):
        """Resets the gym environment, drawing a new ground truth from Poisson 
        prior and returning the first belief representation.

        Args:
            verbose:
              Specifies whether or not verbose logging is used.

        Returns:

          A list of prior means for the default hypotheses, followed by the 
          right boundary of each hypothesis (from left to right). For example, 
          if self.number_hypotheses == 5 then reset() returns

          [
           self.lam/5, 
           self.lam/5, 
           self.lam/5, 
           self.lam/5, 
           self.lam/5,
           1/5,
           2/5,
           3/5,
           4/5,
           5/5,
          ]

          where it is implicit that the left booundary of each hypothesis is the 
          right boundary of the preceeding hypotheis. 
        """
        self.ground_truth = []
        while len(self.ground_truth) == 0:
            self.ground_truth = [np.random.uniform() 
                                for i in range(np.random.poisson(self.lam))]
        self.t = 0
        self.actions = []
        self.disjoint_bins = []
        self.b_lengths = []
        self.observations = []
        self.observations_disjoint_bins = []
        self.observation_lengths = []
        self.all_boundaries = list(np.linspace(0,1,self.num_hypotheses+1))
        self.all_boundaries = self.all_boundaries[1:-1]
        
        self.b_dummies = [self.lam/self.num_hypotheses]*self.num_hypotheses
        self.b_lengths = [1/self.num_hypotheses]*self.num_hypotheses
        return np.array((self.b_dummies + self.all_boundaries + [1] ))

    def step(self, action, verbose=True):
        """Takes a step in the Active Searach Belief MDP.

        Args:
            action:
              The intervals to sense by the full team of agents, as specified
              by the left and right boundary of each sensed interval. For 
              example, if there are 5 agents, then `action` will look like

              [
               torch.Tensor(shape=(2,), dtype=torch.float32),
               torch.Tensor(shape=(2,), dtype=torch.float32),
               torch.Tensor(shape=(2,), dtype=torch.float32),
               torch.Tensor(shape=(2,), dtype=torch.float32),
               torch.Tensor(shape=(2,), dtype=torch.float32),
              ]

              where each torch.Tensor object has shape (2,) and dtype 
              torch.float32
            verbose:
              Specifies whether or not verbose logging is used.              

        Returns:

          A tuple (belief_rep, reward, done, info). If 
          self.number_hypotheses == 5 then `belief_rep` is of the form

          [
           NB1_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB2_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB3_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB4_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB5_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           B1_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B2_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B3_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B4_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B5_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
          ]

          where NBi_hat is an estimate of the posterior mean for the number of
          points in hypothesis bin Bi (e.g. obtained from Gibbs, or EA), and
          Bi_right_boundary is the right boundary of hypothesis bin Bi. 
          Regardless of self.number_hypotheses, `reward` is always a 
          torch.Tensor(shape=(1,), dtype=torch.float32) object. Similarly, 
          `done` is always a torch.Tensor(shape=(1,), dtype=torch.bool). In our
          gym environment, `info` is always null, i.e. info == {}.
        """ 
        pudb.set_trace()
        self.t += 1
        action = torch.sigmoid(torch.tensor(action))
        # Generate the observations and update the hypotheses data structures
        for act in action.reshape((2, self.num_agents)).numpy():
            min_act, max_act = min(act), max(act)     
            length = max_act - min_act       
            if length > 0.0001:
                self.actions.append(act)
                observation = 0
                for point in self.ground_truth:
                    if min_act < point and max_act > point:
                        observation += 1
                observation += np.random.randn()*length*np.sqrt(self.sigma2)
                self.observations.append(observation)
                self.observation_lengths.append(length)

                add_action = True
                for boundary in self.all_boundaries:
                    if abs(min_act - boundary) < 0.0001:
                        add_action = False
                    if abs(max_act - boundary) < 0.0001:
                        add_action = False

                if add_action:
                    self.all_boundaries.append(min_act)
                    self.all_boundaries.append(max_act)

        self.all_boundaries = sorted(self.all_boundaries)                            
        self.disjoint_bins = [[0, self.all_boundaries[0]]]
        self.b_lengths = [abs(self.all_boundaries[0])]
        if abs(self.all_boundaries[0]) < 0.00001:
            print('Point hypothesis!')

        for index, boundary in enumerate(self.all_boundaries[:-1]):
            self.disjoint_bins.append([boundary, 
                self.all_boundaries[index+1]])
            b_len = abs(boundary - self.all_boundaries[index+1])
            if b_len < 0.00001:
                print('Point hypothesis!')
            self.b_lengths.append(b_len)
        self.disjoint_bins.append([self.all_boundaries[-1], 1])    
        b_len = 1 - self.all_boundaries[-1]
        if b_len < 0.00001:
            print('Point hypothesis!')
        self.b_lengths.append(b_len)

        self.observations_disjoint_bins = []
        for dummy_action in self.actions:                                       # if we restrict the number of actions & observations to maintain, then this part will become cheaper
            indices = []
            for b_index, bin_ in enumerate(self.disjoint_bins):
                if (bin_[0] >= min(dummy_action) and  \
                   bin_[0] <= max(dummy_action)) or   \
                   (bin_[1] >=  min(dummy_action) and \
                   bin_[1] <=  max(dummy_action)) or  \
                   (bin_[1] > max(dummy_action) and   \
                   bin_[0] < min(dummy_action)):      \
                   indices.append(b_index)
            if len(indices) == 0:
                print('No overlapping hypotheses!')
            self.observations_disjoint_bins.append(indices)

        self.b_dummies = self.generate_belief_rep()

        # Remove excess hypotheses
        scores = []
        for i in range(len(self.b_dummies)-1):
            new_dummies = copy.deepcopy(self.b_dummies)
            new_b_lengths = copy.deepcopy(self.b_lengths)

            new_entry = new_dummies[i] + new_dummies[i+1]
            new_length = new_b_lengths[i] + new_b_lengths[i+1]

            del new_dummies[i]
            del new_b_lengths[i]

            new_dummies[i] = new_entry
            new_b_lengths[i] = new_length    

            score = np.sum(np.divide(new_dummies,new_b_lengths))                # Use Expectation Density heuristic to remove excess hypotheses
            scores.append(score)
        sorted_indices = np.argsort(scores)
        num_to_drop = len(self.all_boundaries) - (self.num_hypotheses - 1)
        self.all_boundaries = sorted(list(
            np.asarray(self.all_boundaries)[sorted_indices[num_to_drop:]]))
        self.disjoint_bins = [[0, self.all_boundaries[0]]]
        self.b_lengths = [abs(self.all_boundaries[0])]
        for index, boundary in enumerate(self.all_boundaries[:-1]):
            self.disjoint_bins.append([boundary,
                self.all_boundaries[index+1]])
            self.b_lengths.append(
                abs(boundary - self.all_boundaries[index+1]))
        self.disjoint_bins.append([self.all_boundaries[-1], 1])
        self.b_lengths.append(abs(1-self.all_boundaries[-1]))

        self.observations_disjoint_bins = []
        for dummy_action in self.actions:                                       # if we restrict the number of actions & observations to maintain, then this part will become cheaper
            indices = []
            for b_index, bin_ in enumerate(self.disjoint_bins):
                if (bin_[0] >= min(dummy_action) and  \
                   bin_[0] <= max(dummy_action)) or   \
                   (bin_[1] >=  min(dummy_action) and \
                   bin_[1] <=  max(dummy_action)) or  \
                   (bin_[1] > max(dummy_action) and   \
                   bin_[0] < min(dummy_action)):      \
                   indices.append(b_index)
            if len(indices) == 0:
                print('No overlapping hypotheses!')
            self.observations_disjoint_bins.append(indices)

        # Generate the belief representation; could save time here
        self.b_dummies = self.generate_belief_rep()

        # Generate the reward
        reward = self.generate_reward() 
        
        # Generte the done indicator
        done = bool(self.t >= self.num_timesteps)

        # Return the current belief representation, reward, done, info
        return np.array(self.b_dummies + self.all_boundaries + [1]), np.array(reward), np.array(done), {}


    def generate_belief_rep(self):
        """Generates a belief representation suitable for belief-space policy
        optimisation. This code uses three tricks to make inference tractable:
          1. Maintain a maximium number of hypotheses by using the Expectation 
          Density heursitic to eliminate excess hypotheses as described in 
          slides. 
          2. Ignore the conditional dependency between NBi terms
          3. Directly approximate the expectation Gauss-Seidel style by 
          leveraging the monotonic structure of the MAAS problem using the EA 
          algorithm.

        Returns:

          A belief_rep. If self.number_hypotheses == 5 then `belief_rep` is of 
          the form

          [
           NB1_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB2_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB3_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB4_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           NB5_hat: torch.Tensor(shape=(1,), dtype=torch.float32),
           B1_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B2_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B3_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B4_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
           B5_right_boundary: torch.Tensor(shape=(1,), dtype=torch.float32),
          ]

          where NBi_hat is an estimate of the posterior mean for the number of
          points in hypothesis bin Bi (e.g. obtained from Gibbs, or EA), and
          Bi_right_boundary is the right boundary of hypothesis bin Bi. 
        """           
        T = len(self.observations)
        K = len(self.b_lengths)

        A = np.zeros([T,K])
        D = np.zeros(T)

        b = self.observations

        for t in range(T):
            D[t] = 1/(np.sqrt(self.sigma2)*self.observation_lengths[t])
            for k in range(K):
                insert = np.zeros(K)
                insert[self.observations_disjoint_bins[t]] = 1
            A[t] = insert

        D = np.diag(D)

        A_tilde = D @ A
        b_tilde = D @ b

        prior_for_regularisation = np.array([x*self.lam for x in self.b_lengths])

        A_tilde = np.vstack([A_tilde, self.MLE_regularizer*np.eye(K)])
        b_tilde = np.concatenate([b_tilde, self.MLE_regularizer*prior_for_regularisation])

        res = sp.optimize.nnls(A_tilde, b_tilde)[0]
        b_dummies = res

        b_dummies_before = copy.deepcopy(b_dummies)
        b_dummies_after = []
        for b_index in range(len(b_dummies)):
            b_dummies = copy.deepcopy(b_dummies_before)
            mean = 0
            log_p = []
            for dummy_n in range(self.upper_limit_N):
                b_dummies[b_index] = dummy_n
                log_p.append(self.log_f(b_dummies, self.b_lengths, 
                    self.observations, self.observations_disjoint_bins, 
                    self.observation_lengths, self.lam, self.sigma2))

            # Construct appropriate probabilities
            max_log_p = max(log_p)
            for dummy_n in range(self.upper_limit_N):
                if max_log_p - log_p[dummy_n] > self.log_space_resolution:
                    log_p[dummy_n] = max_log_p - self.log_space_resolution

            mean_log_p = np.mean(log_p)
            log_p = [lp - mean_log_p for lp in log_p]

            p = [np.exp(lp) for lp in log_p]
            p /= sum(p)
            for index, dummy_n in enumerate(range(10)):
                mean += dummy_n * p[index]
            b_dummies_after.append(mean)
            b_dummies = copy.deepcopy(b_dummies_before)
        return b_dummies_after

    def log_f(self, b_dummies, b_lengths, observations, 
        observations_disjoint_bins, observation_lengths, lam, sigma2):
        # b_dummies is a vector of dummy variables for the number of points in 
        # {B_i} for all i, ordered left to right b_lengths is a vector of 
        # lengths of {B_i} for all i, ordered left to right observations is a 
        # vector of all the observations we have made, first element is the 
        # first observation in time observations_disjoint_bins is a list, each 
        # elemnts is a list of disjoint bin indices associted with an 
        # observation. First element is the list of bins associated with the 
        # first observatin in time. observation_lengths is a vector of lengths 
        # of {A_i} for all i, ordered forward in time. lam is the PPP 
        # parameter > 0. sigma2 is the observstion noise parameter > 0
        
        K = len(b_dummies)
        T = len(observations)

        log_unnormalised_posterior = 0
        
        for i in range(K):
            log_unnormalised_posterior += self.log_poisson_pmf(int(b_dummies[i]), 
                lam, b_lengths[i])
            
        for t in range(T):
            N = np.sum([ b_dummies[x] for x in observations_disjoint_bins[t] ])
            N *= (observation_lengths[t]) / (np.sum([ b_lengths[x] 
                for x in observations_disjoint_bins[t] ]))            
            log_unnormalised_posterior += self.log_gaussian_density(observations[t], 
                N, observation_lengths[t], sigma2)
        return log_unnormalised_posterior

    def log_poisson_pmf(self, k, lam, length):
        # k is dummy variable (realisation)
        # lam is PPP parameter
        # length is length of interval of PPP
        
        lam = lam*length
        return -lam + (k*np.log(lam)) - np.log(np.math.factorial(k))


    def log_gaussian_density(self, x, N, L, sigma2):
        # x is dummy variable (realisation)
        # N is true number of points in interval
        # L is length of interval
        # sigma2 is variance of noise 
        sigma2 = sigma2*(L**2) 
        sig = np.sqrt(sigma2)
        return -1*np.log(np.sqrt(2*np.math.pi*sig**2)) - 0.5*((x - N)/sig)**2


    def generate_reward(self):
        temp_costs = []
        for cost_iteration in range(self.cost_iterations):
            x_sample = []
            for index, p in enumerate(self.b_dummies):
                num = int(p)
                if np.random.rand() < p - num:
                    num += 1
                for i in range(num):
                    x_sample.append(np.random.uniform(
                        self.disjoint_bins[index][0], 
                        self.disjoint_bins[index][1]))

            if len(x_sample) > 0:
                g = nx.complete_bipartite_graph(self.ground_truth, 
                    x_sample)
                for index, edge in enumerate(g.edges):
                    g[edge[0]][edge[1]]['weight'] = abs(edge[0] - edge[1])
                match = nx.algorithms.bipartite.matching \
                .minimum_weight_full_matching(g)

            cost = abs(len(x_sample) - len(self.ground_truth))
            if len(x_sample) > 0:
                for node1, node2 in match.items():
                    cost += g[node1][node2]['weight']
            temp_costs.append(cost)

        return -1*np.mean(temp_costs)
