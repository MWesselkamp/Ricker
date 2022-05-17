import numpy as np
from scipy.optimize import minimize
import torch

class Ricker:
    """Class for a Ricker model object."""

    def __init__(self):

        """This function specifies the initial state of the model, when an object is created.
            Required args:
                NO (float): initial number of individuals in population.
        """

    def set_prior_observation(self, mu, sigma = 0):

        """
        Specify the normal distribution parameters.
        So far for parameter r only. Distribution normal only.
        :param mu: mean of normal
        :param sigma: scale of normal
        """

        self.mu = mu
        self.sigma = sigma

        return(mu, sigma)

    def set_prior_parameters(self, dict):

        self.r_mean = dict['r_mean']
        self.r_sd = dict['r_sd']

        return dict

    def sample_from_priors(self):

        """
        Function that samples parameter from specified prior distribution.
        So far only for r.
        :return: sample of r.
        """
        N = np.random.normal(self.mu, self.sigma)
        r = np.random.normal(self.r_mean, self.r_sd)

        return(N, r)

    def model(self, N, r, k = 1):
        """
        The model, one time step. Not recursive.
        :param N: Population size at time t.
        :param r: growth rate
        :param k: carrying capacity
        :return: Population size at time t+1
        """
        N = N * np.exp(r * (1 - N / k))
        return(N)

    def model_iterate(self, x_init, r, iterations, k=1):

        """

        :param x_init:
        :param r:
        :param iterations:
        :param k:
        :return:
        """

        x = []
        x.append(x_init)

        for i in range(iterations):
            x.append(x[i] * torch.exp(r * (1 - x[i] / k)))

        x = torch.tensor(x).float()

        return(x)

    def model_simulate(self, samples, iterations, k=1):

        """
        Function used to simulate a bunch of population growth time series with the Ricker model.
        Creates class attribute r, that is a list with growth rates used for simulations. RENAME.
        :param samples: Number of time series to generate. Refers to number of samples from specified prior.
        :param iterations: Number of time steps to simulate.
        :param k: Carrying capacity. Default to 1.
        :return: (array) simulated population growth at every sample of r.
        """

        self.r = []
        simulations = np.zeros((iterations + 1, samples))

        for i in range(samples):
            pars = self.sample_from_priors()
            simulations[0, i] = pars[0]
            r = pars[1]
            self.r.append(r)
            for j in np.arange(iterations):
                simulations[j + 1, i] = self.model(simulations[j, i], r, k)

        return simulations





