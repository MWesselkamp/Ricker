import numpy as np
import scipy.optimize as optim

class Model:

    def __init__(self):
        self.num = np.random.RandomState(100)

    def set_parameters(self, theta, precision=0.1):
        """
        Set true model parameters and precision for sampling under uncertainty.
        :param theta: dict. Model Parameters, i.e. r and sigma.
        :param precision:
        :return:
        """
        self.theta = theta
        self.precision = precision

    def print_parameters(self):
        """
        Prints theta.
        """
        print("True parameter values:")
        try:
            for key, value in self.theta.items():
                print(key, value)
        except:
            print("Model parameters not set!")

    def sample_parameters(self):
        """
        Sample parameters from normal distribution with mean and mean*precision.
        Not used currently.
        """
        pars = []
        for par, mean in self.theta.items():
            pars.append(self.num.normal(mean, mean * self.precision, 1)[0])
        return pars


class Ricker(Model):

    def __init__(self, initial_size, initial_uncertainty):
        """
        Initializes model as the Ricker model (Petchey 2015).
        : initial_size: float. Population size at time 0.
        : intial_uncertainty: float. ?Specify as precision?
        :param stoch: assume a deterministic or stochastic process.
        """
        self.initial_size = initial_size
        self.initial_uncertainty = initial_uncertainty

        super(Ricker, self).__init__()

    def model(self, N, stoch=False):
        """
        With or without stochasticity (Woods 2010).
        :param N: Population size at time step t.
        """
        if not stoch:
            return N * np.exp(self.theta['r'] * (1 - N))
        else:
            return N * np.exp(self.theta['r'] * (1 - N)) + self.theta['sigma'] * self.num.normal(0, 1)

    def model_derivative(self, N):
        """
        Derivative of the Ricker at time step t.
        :param N: Population size at time step t.
        """
        return np.exp(self.theta['r'] - self.theta['r'] * N) * (1 - self.theta['r'] * N)

    def model_iterate(self, iterations, init, obs_error=False, stoch=False):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        iterations: int.
        init: float. Initial Population size.
        obs_error: Add noise to simulated ts?
        """

        timeseries = np.full(iterations, init, dtype=np.float)
        timeseries_derivative = np.full(iterations, self.model_derivative(init), dtype=np.float)

        for i in range(1, iterations):
            timeseries[i] = self.model(timeseries[i - 1], stoch)
            if obs_error:
                timeseries[i] = self.num.poisson(timeseries[i])  # Adjust distribution! No poisson without phi
            timeseries_derivative[i] = self.model_derivative(timeseries[i])

        return timeseries, timeseries_derivative


class Simulation:

    def __init__(self, mod, iterations):
        """
        Requires model object of type Ricker with corresponding class functions and attributes.
        :param mod: class. Model object.
        :param iterations: int. Time steps.
        :param obs_error: assume an observation error.
        """
        self.iterations = iterations
        self.mod = mod
        self.num = self.mod.num

    def sources_of_uncertainty(self, parameters, initial, observation, stoch):

        self.parameters = parameters # parameter error still missing!
        self.initial = initial # check
        self.obs_error = observation # check
        self.stoch = stoch # check

    def simulate(self, ensemble_size = None):
        """
        Calls model class function iterate.
        :return: tuple. simulated timeseries and its derivative.
        """

        if not ensemble_size is None:
            timeseries_array = [None] * ensemble_size
            timeseries_derivative_array = [None] * ensemble_size

            for n in range(ensemble_size):

                if self.initial:
                    initial_condition = self.num.normal(self.mod.initial_size, self.mod.initial_uncertainty)
                    while initial_condition < 0:  # Instead use truncated normal/Half Cauchy
                        initial_condition = self.num.normal(self.mod.initial_size, self.mod.initial_uncertainty)
                else:
                    initial_condition = self.mod.initial_size

                timeseries, timeseries_derivative = self.mod.model_iterate(self.iterations, initial_condition,
                                                                           obs_error = self.obs_error,
                                                                           stoch = self.stoch)
                timeseries_array[n] = timeseries
                timeseries_derivative_array[n] = timeseries_derivative

            return np.array(timeseries_array), np.array(timeseries_derivative_array)

        else:
            if self.initial: # only if inital conditions uncertainty considered
                initial_condition = self.num.normal(self.mod.initial_size, self.mod.initial_uncertainty)
            else:
                initial_condition = self.mod.initial_size
            timeseries, timeseries_derivative = self.mod.model_iterate(self.iterations, initial_condition,
                                                                       obs_error=self.obs_error,
                                                                       stoch = self.stoch)

            return timeseries, timeseries_derivative



# Place this function somewhere else?
def lsq_fit(mod, x_train, bounds = (0, [4.])):
    """
    Fits the ricker model to data in x_train.
    Currently just estimating r. Enable for sigma!
    :param x_train: 1d array, timeseries of true dynamics.
    :param bounds: boundaries for paramter values to sample within
    :return: least squares object
    """

    def fun(pars, x, y):
        mod.theta['r'] = pars[0]
        res = mod.model(x) - y
        return res

    # Initialize parameters randomly
    r_init = np.random.normal(mod.theta['r'], mod.theta['r']*mod.precision, 1)[0]
    p0 = [r_init]

    # Data
    x = x_train[:-1]
    y = x_train[1:]

    lsq_solution = optim.least_squares(fun, p0, bounds=bounds, loss = 'soft_l1', args=(x, y))
    r = lsq_solution.x[0]

    mod.theta_hat = {'r':r, 'sigma':None}

    return lsq_solution


#==========#
# IGNORE!! #
#==========#

import torch
from pyro.infer import MCMC, NUTS
import pyro
import pyro.distributions as dist

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
