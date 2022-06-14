import numpy as np
import torch
from pyro.infer import MCMC, NUTS
import pyro
import pyro.distributions as dist
import pickle

def ricker(N, log_r, phi):
    return np.exp(log_r+np.log(N)-N)*phi

def ricker_derivative(N, log_r, phi):
    return -phi*np.exp(log_r)*np.exp(-N)(N-1)

def iterate_ricker(theta, its, init = None, obs_error = False, stoch=False):

    """
    Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
    :param theta:
    :param its:
    :param obs_error:
    :return:
    """
    seed = 100
    # initialize random number generator
    num = np.random.RandomState(seed)

    log_r = theta['log_r']
    sigma = theta['sigma']
    phi = theta['phi']

    # Initialize the time-series
    if not init is None:
        timeseries_obs_size = np.full((its), init, dtype=np.float)
        timeseries_true_size = np.full((its), init, dtype=np.float)
    else:
        timeseries_obs_size = np.zeros((its), dtype=np.float)
        timeseries_true_size = np.ones((its), dtype=np.float)

    for i in range(1, its):
        if stoch:
            timeseries_true_size[i] = np.exp(log_r+np.log(timeseries_true_size[i-1])-timeseries_true_size[i-1]+sigma*num.normal(0, 1))
            if obs_error:
                timeseries_obs_size[i] = num.poisson(phi*timeseries_true_size[i])
            else:
                timeseries_obs_size[i] = phi * timeseries_true_size[i]
        else:
            timeseries_true_size[i] = np.exp(log_r+np.log(timeseries_true_size[i-1])-timeseries_true_size[i-1])
            if obs_error:
                timeseries_obs_size[i] = num.poisson(phi*timeseries_true_size[i])
            else:
                timeseries_obs_size[i] = phi * timeseries_true_size[i]

    return timeseries_true_size, timeseries_obs_size

def ricker_simulate(samples, its, theta, init = None, obs_error = False, stoch=False):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        """
        timeseries_array_obs = [None]*samples
        timeseries_array_true = [None]*samples

        if init is not None:
            seed = 100
            # initialize random number generator
            num = np.random.RandomState(seed)

        for n in range(samples):

            init_sample = num.normal(init[0], init[1])
            while init_sample < 0:
                init_sample = num.normal(init[0], init[1])

            timeseries_true_size, timeseries_obs_size = iterate_ricker(theta, its, init_sample, obs_error, stoch)
            timeseries_array_obs[n] = timeseries_obs_size
            timeseries_array_true[n] = timeseries_true_size

        return np.array(timeseries_array_obs)

# Model as standard class object
class Ricker:

    """Class for a Ricker model object."""

    def __init__(self, stoch, N0_real, r_real, dyn_real):

        """This function specifies the initial state of the model, when an object is created.
            Required args:
                NO (float): initial number of individuals in population.
        """

        self.stoch = stoch
        self.N0_real = N0_real
        self.r_real = r_real
        self.dyn_real = dyn_real

    def model_iterate(self, N0, r, iterations, sigma = None):

        """
        :param x_init:
        :param r:
        :param iterations:
        :param k:
        :return:
        """

        x = np.zeros((iterations + 1, len(r)))
        x[0, :] = N0
        print("Initial values", x[0, :])
        print("Paramvalues", r)
        for i in range(iterations):
            if not sigma is None:
                x[i+1, :] = np.random.normal(x[i,:] * np.exp(np.multiply(r, (1 - x[i,:] / self.k))), sigma)
            else:
                x[i+1, :] = x[i,:] * np.exp(np.multiply(r, (1 - x[i,:] / self.k)))
        return x

    # Model in Pyro
    def model_pyro(self):

        sigma = pyro.sample('sigma', dist.Uniform(0, 0.1))
        r = pyro.sample('r', dist.Uniform(1.0, 1.6))

        N0 = self.dyn_real.flatten()[0]

        # At the moment we assume a constant growth rate over time. To change this, loop here over time steps.
        # https: // www.youtube.com / watch?v = tw0cSm7TElE (minute 19)
        preds = []
        preds.append(N0)
        for i in range(self.dyn_real.shape[0]):
            preds.append(preds[i] * torch.exp(r * (1 - preds[i] / self.k)))
        #preds = torch.tensor(preds).float()
        # Change to poisson likelihood with scale parameter as in woods(2010)

        with pyro.plate('y'):
            y = pyro.sample('y', dist.Normal(preds, sigma), obs=X)

        return y

    def fit_model_pyro(self):

        # Run inference in Pyro
        nuts_kernel = NUTS(self.model_pyro, max_tree_depth=5)
        mcmc = MCMC(nuts_kernel, num_samples=300, warmup_steps=50, num_chains=2)
        mcmc.run(torch.tensor(self.dyn_real))

        print(mcmc.summary())
        mcmc.diagnostics()
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples



