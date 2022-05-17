
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('module://backend_interagg')
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath("/Users/Marieke_Wesselkamp/PycharmProjects/Ricker/venv"))

from model import Ricker

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
# from pyro.optim import Adam, ClippedAdam

#=========================#
# Simulate reference data #
#=========================#

r_real = 1.6
sd_real = 0.01 # assume no uncertainty on observations.
N0 = 0.8
sigma = 0.05

mod = Ricker() # Create class instance
mod.set_prior_observation(mu = N0, sigma=sigma) # set prior for observation error
mod.set_prior_parameters({'r_mean':r_real, 'r_sd':sd_real}) # set prior for parameters

real_dynamics = mod.simulate(samples = 10, iterations = 45)

def plot_growth(real_dynamics):

    plt.plot(real_dynamics)
    plt.show()

# Split real dynamics into train and test.
real_train = real_dynamics[:30, :]
real_test = real_dynamics[31:, :]

# Save to input folder.

#====================#
# Fit the model data #
#====================#

# Set up the model in Pyro
def model_pyro(X, obs=None):

    # Specify these as parameters instead of random variables?
    sigma = pyro.sample('sigma', dist.HalfCauchy(0.1)) #dist.Normal(0, mod.sigma)) #prior for observation error
    r = pyro.sample('r', dist.Normal(mod.r_mean, mod.r_sd)) #prior for parameters

    its = X.shape[0]-1

    # At the moment we assume a constant growth rate over time. To change this, loop here over time steps.
    # https: // www.youtube.com / watch?v = tw0cSm7TElE (minute 19)
    preds = mod.model_iterate(X[0], r, iterations=its)

    y = pyro.sample('y', dist.Normal(preds, sigma), obs=X)

    return y

# prep data for Pyro model
X = torch.tensor(real_train.flatten()).float()

# Run inference in Pyro
nuts_kernel = NUTS(model_pyro)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100, num_chains=1)
mcmc.run(X)

mcmc.summary()
mcmc.diagnostics()
posterior_samples = mcmc.get_samples()

# Save the model and the model fit, i.e. posterior distributions.
# IDEA: Save posterior distributions in model class object.

def plot_posterior(ps):

    """
    Write a function, that displays posterior distributions.
    Save plot to plots folder.
    Move it to a visualizations script.
    :param ps:
    :return:
    """


# Press the green button in the gutter to run the script.
# Produce results

if __name__ == '__main__':
    plot_growth(real_dynamics)


