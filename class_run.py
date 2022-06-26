import dynamics
import modelclass
import utils

#==========================================#
# The Ricker model for population dynamics #
#==========================================#

# Set hyperparameters.
its = 100 # number of iterations to simulate
train_size = 50 # splits
test_index = 51
initial_population_mean = 0.8 # inital population size
initial_uncertainty = 0 # No uncertainty in real dynamics (Petchey 2015)
ensemble_size = 10
ensemble_uncertainty = 1e-5

theta = {'r':2.9, 'sigma':0.3} # true parameter values (Petchey 2015)

# Initalize model
ricker = modelclass.Ricker(initial_population_mean, initial_uncertainty)
ricker.set_parameters(theta = theta)
ricker.print_parameters()

# Simulate 'true' dynamics
simulator = modelclass.Simulation(ricker, iterations=its)
ts_true, ts_true_derivative = simulator.simulate_single()

# Simulate ensemble with perfect model knowledge
perfect_ensemble, perfect_ensemble_derivative = simulator.simulate_ensemble(ensemble_size, ensemble_uncertainty)

#==============================#
# Fit model with Least squares #
#==============================#

x_train = ts_true[:train_size]
x_test = ts_true[test_index:]

lsqfit = modelclass.lsq_fit(ricker, x_train) # Changes theta in model object! Find other solution.
# lsq fit standard errors on estimates?

#===================#
# Forecast horzions #
#===================#

# Historic mean
historic_mean, historic_var = utils.historic_mean(x_test, x_train)

# Predictability time with Lyapunov time.
lyapunovs = dynamics.lyapunovs(perfect_ensemble_derivative)
lyapunovs_stepwise = dynamics.lyapunovs(perfect_ensemble_derivative, stepwise=True)