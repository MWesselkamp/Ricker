import numpy as np
import modelclass

# Set hyperparameters.
uncertainties = {"its":100, "parameters":False, "initial":True,"observation":False,"stoch":False}
hp = {"iterations":100, "initial_size": 0.8, "initial_uncertainty": 1e-5, "ensemble_size": 50}
theta = {'r':2.9, 'sigma':0.3} # true parameter values (Petchey 2015)

# Initalize model
ricker = modelclass.Ricker(uncertainties)
ricker.set_parameters(theta)
simu = ricker.simulate(hp)

#===========================================#
# Forecast horizon with signal noise ratio. #
#===========================================#