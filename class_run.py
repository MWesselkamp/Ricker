import modelclass

its = 100
train_size = 50
test_index = 51
initial_population_mean = 0.8
initial_uncertainty = 0 # No uncertainty in real time series (Petchey)
ensemble_size = 10
ensemble_uncertainty = 1e-5

theta = {'r':2.9, 'sigma':0.3}

ricker = modelclass.Ricker(initial_population_mean, initial_uncertainty)
ricker.set_parameters(theta = theta)
ricker.print_parameters()

# Simulate true dynamics
simulator = modelclass.Simulation(ricker, iterations=its)
ts_true, ts_true_derivative = simulator.simulate_single()

# Simulate ensemble with perfect model knowledge
perfect_ensemble, perfect_ensemble_derivative = simulator.simulate_ensemble(ensemble_size, ensemble_uncertainty)


#==============================#
# Fit model with Least squares #
#==============================#

x_train = ts_true[:train_size]
x_test = ts_true[test_index:]

lsqfit = modelclass.lsq_fit(ricker, x_train) # Changes theta in model object.






# timeseries_lyapunovs_array = np.log(abs(timeseries_derivative_array))
# lyapunovs = np.mean(np.array(timeseries_lyapunovs_array), axis=1)