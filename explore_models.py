import numpy as np
import models
import utils

uncertainties = {"parameters":False, "initial":True,"observation":False,"stoch":False}

#=======================================#
# Multi-species model with T-dependence #
#=======================================#

# Set hyperparameters.
hp_rm = {"iterations":50, "initial_size": (0.8, 0.9), "initial_uncertainty": 1e-5, "ensemble_size": 50}
# Set parameters
theta_rm = {'alpha':1, 'beta':0.1, 'gamma': 1, 'delta':0.1, 'sigma':None} # random values
theta_rm_upper = {'ax': 0.95, 'bx': 1.5, 'cx': 1.2, 'ay':1.0, 'by': 1.2, 'cy':1.1} # random values
T = utils.simulate_T(hp_rm['iterations'], add_trend=True)

ricker_multi_t = models.Ricker_Multi_T(uncertainties)
ricker_multi_t.set_parameters(theta_rm, theta_rm_upper)
simu = ricker_multi_t.simulate(hp_rm, derive=False, ex=T)
x = simu["ts"]
ricker_multi_t.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))

#======================#
# Multi-species model. #
#======================#


# Set hyperparameters.
hp_rm = {"iterations":50, "initial_size": (0.8, 0.9), "initial_uncertainty": 1e-5, "ensemble_size": 50}
# Set parameters
theta_rm = {'lambda_a': np.exp(2.9), 'lambda_b': np.exp(2.9), 'alpha':1, 'beta':0.1, 'gamma': 1, 'delta':0.1, 'sigma':None} # true parameter values (Petchey 2015)

ricker_multi = models.Ricker_Multi(uncertainties)
ricker_multi.set_parameters(theta_rm)
simu = ricker_multi.simulate(hp_rm, derive=False)
x = simu["ts"]
ricker_multi.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))


#=======================#
# Single-species model. #
#=======================#

# Set hyperparameters.
hp_r = {"iterations":50, "initial_size": 0.8, "initial_uncertainty": 1e-5, "ensemble_size": 50}

# Set parameters
theta_r1 = {'r':2.9, 'sigma':None} # true parameter values (Petchey 2015)
# Initialize model 1 (Petchey 2015)
ricker = models.Ricker_1(uncertainties)
ricker.set_parameters(theta_r1)
simu = ricker.simulate(hp_r)
x = simu["ts"]
ricker.visualise(np.transpose(x))


# Set parameters
theta_r2 = {'lambda':np.exp(2.9), 'alpha':1, 'sigma':None} # lambda = exp(r)
# Initialize model 2 (Otto 2015)
ricker = models.Ricker_2(uncertainties)
ricker.set_parameters(theta_r2)
simu = ricker.simulate(hp_r)
x = simu["ts"]
ricker.visualise(np.transpose(x))

