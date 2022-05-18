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

        self.stoch = False

    def set_priors(self, initial_conditions, parameters, process_error = None):

        """
        Specify the uncertainties that should be propagated.
        So far for parameter r only. Distribution normal only.
        :param mu: mean of normal
        :param sigma: scale of normal
        """

        self.N0_mean = initial_conditions['N0_mean']
        self.N0_sd = initial_conditions['N0_sd']

        self.k = parameters['k']
        self.r_mean = parameters['r_mean']
        self.r_sd = parameters['r_sd']

        if not process_error is None:
            self.mu = process_error['mu']
            self.sigma = process_error['sigma']
            self.stoch = True

        #return(mu, sigma)

    def sample_from_priors(self):

        """
        Function that samples parameter from specified prior distribution.
        So far only for r.
        :return: sample of r.
        """
        N = np.random.normal(self.N0_mean, self.N0_sd)
        r = np.random.normal(self.r_mean, self.r_sd)

        return(N, r)

    def model(self, N, r):
        """
        The model, one time step. Not recursive.
        :param N: Population size at time t.
        :param r: growth rate
        :param k: carrying capacity
        :return: Population size at time t+1
        """
        N = N * np.exp(r * (1 - N / self.k))
        return(N)

    def model_iterate(self, N0, r, iterations):

        """

        :param x_init:
        :param r:
        :param iterations:
        :param k:
        :return:
        """

        x = []
        x.append(N0)

        for i in range(iterations):
            x.append(x[i] * torch.exp(r * (1 - x[i] / self.k)))

        x = torch.tensor(x).float()

        return(x)

    def model_simulate(self, samples, iterations):

        """
        Function used to simulate a bunch of population growth time series with the Ricker model.
        Creates class attribute r, that is a list with growth rates used for simulations. RENAME.
        :param samples: Number of time series to generate. Refers to number of samples from specified prior.
        :param iterations: Number of time steps to simulate.
        :param k: Carrying capacity. Default to 1.
        :return: (array) simulated population growth at every sample of r.
        """

        self.r_samples = []
        simulations = np.zeros((iterations + 1, samples))

        for i in range(samples):

            pars = self.sample_from_priors()

            if self.stoch:
                simulations[0, i] = np.random.normal(pars[0], self.sigma)
            else:
                simulations[0, i] = pars[0]

            r = pars[1]
            self.r_samples.append(r)

            for j in np.arange(iterations):
                if self.stoch:
                    simulations[j + 1, i] = self.model(simulations[j, i], r)
                else:
                    simulations[j + 1, i] = np.random.normal(self.model(simulations[j, i], r), self.sigma)

        return simulations





