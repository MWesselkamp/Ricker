import modelclass
import visualisations
import numpy as np

# Set hyperparameters.
its = 100 # number of iterations to simulate
train_size = 50 # splits
test_index = 51
initial_size = 0.8 # inital population size
initial_uncertainty = 1e-5 # No uncertainty in real dynamics! (Petchey 2015)
ensemble_size = 50

theta_ricker = {'r':2.9, 'sigma':0.3}
theta_hassell = {'alpha':0.8, 'lambda':1.2, 'theta':0.1, 'sigma':0.3}

# Initalize model
ricker = modelclass.Ricker(initial_size, initial_uncertainty)
ricker.set_parameters(theta = theta_ricker)

hassell = modelclass.Hassell(initial_size, initial_uncertainty)
hassell.set_parameters(theta = theta_hassell)
simulator_hassell = modelclass.Simulation(hassell, iterations=its) # Create a simulator object
# To simulate the baseline dynamics, all error sources are false
simulator_hassell.sources_of_uncertainty(parameters=False,
                                 initial = True,
                                 observation = False,
                                 stoch = False)
perfect_ensemble, perfect_ensemble_derivative = simulator_hassell.simulate(ensemble_size)
vizualisations.plot_trajectories(perfect_ensemble, its, np.mean(perfect_ensemble, axis=0))
