import numpy as np
import matplotlib.pyplot as plt
from model import Ricker

#=========================#
# Simulate reference data #
#=========================#

r_real = 1.5
sd_real = 0.1
N0_mean = 0.8
N0_sd = 0.05
mu = 0
sigma = 0.05

mod = Ricker() # Create class instance

obs_error = {'N0_mean':N0_mean, 'N0_sd':N0_sd}
param_prior = {'r_mean':r_real, 'r_sd':sd_real, 'k':20}
process_error = {'mu': mu, 'sigma': sigma}

mod.set_priors(obs_error, param_prior, process_error)
real_dynamics = mod.model_simulate(samples = 1, iterations = 50)

def plot_growth(real_dynamics):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(real_dynamics)
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
    fig.show()
    fig.savefig('plots/real_dynamics.png')

# Split real dynamics into train and test.
real_train = real_dynamics[:30, :]
real_test = real_dynamics[31:, :]


if __name__ == '__main__':

    np.savetxt("data/realdyn_train.csv", real_train)
    np.savetxt("data/realdyn_test.csv", real_test)
    plot_growth(real_dynamics)
