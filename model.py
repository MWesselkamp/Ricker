import numpy as np
import torch
from pyro.infer import MCMC, NUTS
import pyro
import pyro.distributions as dist
import scipy.optimize as optim

def historic_mean(x_test, x_train, length = 'test'):

    if length=='test':
        length = x_test.shape[0]
    historic_mean = np.full((length), np.mean(x_train), dtype=np.float)
    historic_var = np.full((length), np.std(x_train), dtype=np.float)
    return historic_mean, historic_var

def ricker(N, r, sigma=None, seed = 100):

    """
    Ricker according to Petchey (2015) or ecolmod R package for a carrying capacity k of 1.
    :param N:
    :param log_r:
    :param phi:
    :param sigma:
    :param seed:
    :return:
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
    :param theta:
    :param its:
    :param obs_error:
    :return:
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
        timeseries_log_abs_array = [None] * samples
        # initialize random number generator
        num = np.random.RandomState(seed)

        for n in range(samples):

            init_sample = num.normal(init[0], init[1])
            while init_sample < 0:
                init_sample = num.normal(init[0], init[1])

            timeseries, timeseries_log_abs = iterate_ricker(theta, its, init_sample, obs_error)
            timeseries_array[n] = timeseries
            timeseries_log_abs_array[n] = timeseries_log_abs

        lyapunovs = np.mean(np.array(timeseries_log_abs_array), axis=1)

        if samples != 1:
            return np.array(timeseries_array), lyapunovs
        else:
            return np.array(timeseries_array)[0], lyapunovs


class Model:

    def __init__(self):
        self.num = np.random.RandomState(100)

    def set_parameters(self, theta):
        self.theta = theta

    def set_boundaries(self, theta_bounds):
        self.theta_bounds = theta_bounds

    def print_parameters(self):
        print("Model parameters")
        try:
            for key, value in self.theta.items():
                print(key, value)
        except:
            print("Model parameters are not set.")

    def sample_parameters(self):

        print("Sampling parameters with default standard dev of 0.5")
        pars = []
        for par, mean in self.theta.items():
            pars.append(self.num.normal(mean, 0.5, 1)[0])
        return pars

    def model_fit_lsq(self):

        def fun(pars, x, y):
            res = self.model(x, pars) - y
            return res

        pars = self.sample_parameters()
        lsq_fit = optim.least_squares(fun, pars, bounds=self.theta_bounds, loss='soft_l1', args=(self.x, self.y))
        self.theta_hat = dict(zip(self.theta.keys(),lsq_fit.x))

        return lsq_fit

class Ricker(Model):

    def __init__(self, stoch=False):
        super(Ricker, self).__init__()
        self.stoch = stoch

    def split_data(self, x_train):
        self.x = x_train[:-1]
        self.y = x_train[1:]

    def model(self, N, pars=None):

        if pars is None:
            log_r = self.theta['log_r']
            sigma = self.theta['sigma']
            phi = self.theta['phi']
        else:
            log_r = pars[0]
            sigma = pars[1]
            phi = pars[2]

        if self.stoch:
            return np.exp(log_r + np.log(N) - N + sigma * self.num.normal(0, 1)) * phi
        else:
            return np.exp(log_r + np.log(N) - N) * phi

    def initalize(self, initial_size, initial_uncertainty):
        self.initial_size = initial_uncertainty
        self.initial_uncertainty = initial_size, initial_uncertainty

    def model_derivative(self, N):
        return -self.theta['phi'] * np.exp(self.theta['log_r']) * np.exp(-N) * (N - 1)



class Simulations:

    def __init__(self, model):

        self.model = model
        self.num = model.num
        self.obs_error = False

    def simulate(self, samples, iterations):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        """
        self.iterations = iterations
        timeseries_array = [None] * samples
        timeseries_log_abs_array = [None] * samples
        # initialize random number generator

        for n in range(samples):

            init_sample = self.num.normal(self.model.initial[0], self.model.initial[1])
            while init_sample < 0:
                init_sample = self.num.normal(self.model.initial_size[0], self.model.initial_uncertainty[1])

            timeseries, timeseries_log_abs = self.iterate(init_sample)
            timeseries_array[n] = timeseries
            timeseries_log_abs_array[n] = timeseries_log_abs

        lyapunovs = np.mean(np.array(timeseries_log_abs_array), axis=1)

        if samples != 1:
            return np.array(timeseries_array), lyapunovs
        else:
            return np.array(timeseries_array)[0], lyapunovs


    def iterate(self, init):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        :param theta:
        :param its:
        :param obs_error:
        :return:
        """

        timeseries = np.full((self.iterations), init, dtype=np.float)

        timeseries_log_abs = np.zeros((self.iterations), dtype=np.float)
        timeseries_log_abs[0] = np.log(abs(self.model.derivative(init)))

        for i in range(1, self.iterations):

            timeseries[i] = self.model.model(timeseries[i - 1])
            if self.obs_error:
                timeseries[i] = self.num.poisson(timeseries[i])

            timeseries_log_abs[i] = np.log(abs(self.model.derivative(timeseries[i])))

        return timeseries, timeseries_log_abs



# Model as standard class object
class Ricker_1:

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



