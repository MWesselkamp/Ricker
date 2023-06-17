from vizualisations import plot_posterior
from vizualisations import plot_trajectories
import pickle
import numpy as np

import torch
from pyro.infer import MCMC, NUTS
import pyro
import pyro.distributions as dist

def ricker(N, r, sigma=None, seed = 100):

    """
    Ricker according to Petchey (2015) or ecolmod R package for a carrying capacity k of 1.
    """

    num = np.random.RandomState(seed)
    if sigma is None:
        return N*np.exp(r*(1-N))
    else:
        return N*np.exp(r*(1-N)) + sigma*num.normal(0, 1)

def ricker_derivative(N, r):
    return np.exp(r-r*N)*(1-r*N)

def iterate_ricker(theta, its, init, obs_error = False, seed = 100):

    """
    Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.

    """
    r = theta['r']
    sigma = theta['sigma']

    num = np.random.RandomState(seed)

    # Initialize the time-series
    timeseries = np.full((its), init, dtype=np.float)
    # Initalize timeseries for lyapunov exponent
    timeseries_log_abs = np.zeros((its), dtype=np.float)
    timeseries_log_abs[0] = np.log(abs(ricker_derivative(init, r)))

    for i in range(1, its):

        timeseries[i] = ricker(timeseries[i-1], r, sigma)
        if obs_error:
            # this can't be a possion distribution! Change this to something else.
            timeseries[i] = num.poisson(timeseries[i])

        timeseries_log_abs[i] = np.log(abs(ricker_derivative(timeseries[i], r)))

    return timeseries, timeseries_log_abs

def ricker_simulate(samples, its, theta, init, obs_error = False, seed=100):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        """
        timeseries_array = [None]*samples
        lyapunovs_array = [None] * samples
        # initialize random number generator
        num = np.random.RandomState(seed)

        for n in range(samples):

            init_sample = num.normal(init[0], init[1])
            while init_sample < 0:
                init_sample = num.normal(init[0], init[1])

            timeseries, timeseries_log_abs = iterate_ricker(theta, its, init_sample, obs_error)
            timeseries_array[n] = timeseries
            lyapunovs_array[n] = timeseries_log_abs

        if samples != 1:
            return np.array(timeseries_array), np.array(lyapunovs_array)
        else:
            return np.array(timeseries_array)[0], np.array(lyapunovs_array)

# Model as standard class object
class Ricker_pyro:

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


if __name__ == '__main__':

    r_real = 1.5
    sd_real = 0.001
    N0_mean = 0.8
    N0_sd = 0
    mu = 0
    sigma = 0.005

    params = {'N0_mean': N0_mean, 'N0_sd': N0_sd, 'r_mean': r_real, 'r_sd': sd_real, 'k': 1}

    x, trues = ricker_simulate(1, 50, params, stoch = False)
    np.savetxt("data/realdynamics.csv", x, delimiter=',')

    x_train = x[:, :30]
    x_test = x[:, 31:]

    mod = Ricker_pyro()  # Create class instance
    mcmc, post_samples = mod.fit_model_pyro(x_train)
    fit = open('../results/fit.pkl', 'wb')
    pickle.dump(post_samples, fit)
    fit.close()

    post_samples = open("../results/fit.pkl", "rb")
    post_samples = pickle.load(post_samples)

    r_posterior = post_samples['r'].numpy()
    sigma_posterior = post_samples['sigma'].numpy()

    plot_posterior(post_samples['r'])
    plot_posterior(post_samples['sigma'])

    trajectories = mod.model_iterate(N0_mean, r_posterior, iterations=50, sigma = np.mean(sigma_posterior))
    plot_trajectories(np.transpose(trajectories), x)
