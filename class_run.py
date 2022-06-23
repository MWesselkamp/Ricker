import model

theta_true = {'log_r':2.9, 'sigma':None, 'phi':10}
its = 100 # Petchey 2015
train_size = 50
test_index = 51

initial_population_mean = 0.8
initial_uncertainty = 0 # No uncertainty in real time series (Petchey)

ensemble_size = 10
ensemble_uncertainty = 1e-5

timeseries, timeseries_lyapunovs = model.ricker_simulate(1, its, theta_true,
                                             init = (initial_population_mean, initial_uncertainty),
                                             obs_error=False)

x_train = timeseries[:train_size]
x_test = timeseries[test_index:]

theta= {'log_r':2.9, 'sigma':0.3, 'phi':10}
ricker = model.Ricker()
ricker.set_parameters(theta = theta)
ricker.set_boundaries(theta_bounds=(0, [4., 2.0, 15]))
ricker.print_parameters()

#ricker.sample_parameters()
ricker.split_data(x_train)
lsq_solution = ricker.model_fit_lsq(x_train)