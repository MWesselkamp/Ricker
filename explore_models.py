import numpy as np
import models
import utils

uncertainties = {"parameters":False, "initial":True,"observation":True,"stoch":False}

#=======================================#
# Multi-species model with T-dependence #
#=======================================#

# Set hyperparameters.
# Say we sample once a week for eight years
hp_rm = {"iterations":416, "initial_size": (20, 20), "initial_uncertainty": 1e-1, "ensemble_size": 10}
# Set parameters
# alpha/gamma: Inverse of the carrying capacity. Use initial size to start at equilibrium. Here 20.
# beta/delta: Inverse of interaction strengh - the higher the weaker the interaction.
# ax/ay: growth rate.
# b/c: Temperature dependency.
theta_rm = {'alpha':1/20, 'beta':35, 'gamma': 1/20, 'delta':45, 'sigma':None} # random values
theta_rm_upper = {'ax': np.exp(1.8), 'bx': 1.8, 'cx': 2.2, 'ay':np.exp(1.8), 'by': 1.0, 'cy':2.1} # random values
T = utils.simulate_T(hp_rm['iterations'], add_trend=False, add_noise=True)

ricker_multi_t = models.Ricker_Multi_T(uncertainties, set_seed=False)
ricker_multi_t.set_parameters(theta_rm, theta_rm_upper)
simu = ricker_multi_t.simulate(hp_rm, derive=False, ex=T)
x = simu["ts"]
ricker_multi_t.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))

#======================#
# Multi-species model. #
#======================#


# Set hyperparameters.
hp_rm = {"iterations":416, "initial_size": (20, 20), "initial_uncertainty": 1e-2, "ensemble_size": 50}
# Set parameters
theta_rm = {'lambda_a': np.exp(2.0), 'lambda_b': np.exp(2.0), 'alpha':1/20, 'beta':30, 'gamma': 1/20, 'delta':30, 'sigma':None} # true parameter values (Petchey 2015)

ricker_multi = models.Ricker_Multi(uncertainties)
ricker_multi.set_parameters(theta_rm)
simu = ricker_multi.simulate(hp_rm, derive=False)
x = simu["ts"]
ricker_multi.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))


#=======================#
# Single-species model. #
#=======================#

# Set hyperparameters.
hp_r = {"iterations":416, "initial_size": (20), "initial_uncertainty": 1e-2, "ensemble_size": 50}

# Set parameters
theta_r1 = {'lambda': np.exp(2.0), 'alpha':1/20, 'sigma':None}
ricker = models.Ricker_Single(uncertainties, set_seed=False)
ricker.set_parameters(theta_r1)
simu = ricker.simulate(hp_r)
x = simu["ts"]
ricker.visualise(np.transpose(x))


#========================================#
# Single-species model with Temperature. #
#========================================#

# Set parameters
theta_r1 = {'alpha':1/20, 'sigma':None} # lambda = exp(r)
theta_r1_upper = {'ax': np.exp(2.0), 'bx': 2.5, 'cx': 2.2}
T = utils.simulate_T(hp_r['iterations'], add_trend=False)

# Initialize model 2 (Otto 2015)
ricker = models.Ricker_Single_T(uncertainties, set_seed=False)
ricker.set_parameters(theta_r1, theta_r1_upper)
simu = ricker.simulate(hp_r, derive=False, ex=T)
x = simu["ts"]
ricker.visualise(np.transpose(x))