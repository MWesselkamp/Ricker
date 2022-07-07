import dynamics
import modelclass
import utils
import proficiency_metrics
import vizualisations
import numpy as np

#==========================================#
# The Ricker model for population dynamics #
#==========================================#

# Set hyperparameters.
its = 100 # number of iterations to simulate
train_size = 50 # splits
test_index = 51
initial_size = 0.8 # inital population size
initial_uncertainty = 1e-5 # No uncertainty in real dynamics! (Petchey 2015)
ensemble_size = 50

theta = {'r':2.9, 'sigma':0.3} # true parameter values (Petchey 2015)

# Initalize model
ricker = modelclass.Ricker(initial_size, initial_uncertainty)
ricker.set_parameters(theta = theta)
ricker.print_parameters()


simulator = modelclass.Simulation(ricker, iterations=its) # Create a simulator object
# To simulate the baseline dynamics, all error sources are false
simulator.sources_of_uncertainty(parameters=False,
                                 initial = False,
                                 observation = False,
                                 stoch = False)
ts_true, ts_true_derivative = simulator.simulate() # Create a single dynamic under perfect conditions

# Simulate ensemble with perfect model knowledge by disturbing initial conditions slightly
perfect_ensemble, perfect_ensemble_derivative = simulator.simulate(ensemble_size) # resulting trajectories are all the same!

# Now disturb initial conditions slightly. Doing so, set initial to True. This will use initial_uncertainty in ricker as sd.
# For reproducing Spring & Ilynia, disturb only after first year of initialization!
simulator.initial = True
perfect_ensemble_d, perfect_ensemble_derivative_d = simulator.simulate(ensemble_size) # resulting trajectories now slightly differ!

#===============================#
# The Lyapunov forecast horizon #
#===============================#

# Choose a forecast proficiency. Here: Absolute difference to truth.
abs_diff, abs_diff_mean = proficiency_metrics.absolute_difference(ts_true, perfect_ensemble_d, mean = True)
vizualisations.FP_absdifferences(abs_diff, abs_diff_mean, its)
t_stats, p_vals = proficiency_metrics.t_statistic(abs_diff, initial_uncertainty) # Where p-value is smaller than threshold.

# Predictability time with Lyapunov time.
lyapunovs = dynamics.lyapunovs(perfect_ensemble_derivative_d)
lyapunovs_stepwise = dynamics.lyapunovs(perfect_ensemble_derivative_d, stepwise=True)

Delta_range = np.linspace(initial_uncertainty, abs_diff.max(), 40)

predicted_efh = np.array([dynamics.lyapunov_efh(lyapunovs, Delta, initial_uncertainty) for Delta in Delta_range])
vizualisations.plot_LE_efh_along_Delta(Delta_range, predicted_efh)

#=================================================#
# Forecast horizon follwoing Spring & Ilyina 2018 #
#=================================================#

# Required: Bootstrap perfect ensemble function:
# Required: Skill metric (default: Pearsons r.)

#for i in range(ensemble_size):
#    leftout = perfect_ensemble_d[i,:] # assign the leftout trajectory
#    rest = np.delete(perfect_ensemble_d, i, 0) # assign the rest

#=================================================#
# Forecast horizon follwoing Séférian et al 2018 #
#=================================================#

#==============================#
# Fit model with Least squares #
#==============================#

x_train = ts_true[:train_size]
x_test = ts_true[test_index:]

# Historic mean
historic_mean, historic_var = utils.historic_mean(x_test, x_train)

lsqfit = modelclass.lsq_fit(ricker, x_train) # Changes theta in model object! Find other solution.
# lsq fit standard errors on estimates?