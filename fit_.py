import matplotlib
matplotlib.use('module://backend_interagg')

import vizualisations
import utils
import model

import numpy as np
import scipy.optimize as optim

seed = 100
# initialize random number generator

#===========================#
# Simulate true time series #
#===========================#

theta_true = {'log_r':2.9, 'sigma':None, 'phi':10}
its = 100 # Petchey 2015
train_size = 50
test_index = 51

initial_population_mean = 0.8
initial_uncertainty = 0 # No uncertainty in real time series (Petchey)

ensemble_size = 10
ensemble_uncertainty = 0.01

timeseries, timeseries_log_abs = model.ricker_simulate(1, its, theta_true,
                                             init = (initial_population_mean, initial_uncertainty),
                                             obs_error=False, stoch=False)
timeseries = timeseries[0]
lyapunov = np.mean(timeseries_log_abs)

x_train = timeseries[:train_size]
x_test = timeseries[test_index:]

#=======================#
# Fit Ricker to x_train.#
#=======================#

## 1. Approach for data with purely deterministic model: Least squares.

# function to minimize: Residuals
def fun(pars, x, y):
    res = model.ricker(x, pars[0], pars[1]) - y
    return res

# Initialize parameters randomly
log_r_init = np.random.normal(theta_true['log_r'], 0.5, 1)[0]
phi_init = np.random.normal(theta_true['phi'], 0.5, 1)[0]
p0 = [log_r_init, phi_init]

# Set min bound 0 on all coefficients, and set different max bounds for each coefficient
bounds = (0, [4., 15])
# Data
x = x_train[:-1]
y = x_train[1:]

lsq_solution = optim.least_squares(fun, p0, bounds=bounds, loss = 'soft_l1', args=(x, y))
log_r = lsq_solution.x[0]
phi = lsq_solution.x[1]
theta_hat = {'log_r':log_r, 'sigma':None, 'phi':phi}

#=======================#
# Forecast time series  #
#=======================#

# 1. Forecast with historic mean
historic_mean = np.full((x_test.shape[0]), np.mean(x_train), dtype=np.float)
historic_var = np.full((x_test.shape[0]), np.std(x_train), dtype=np.float)
print(historic_mean, historic_var)

# 2. Forecast with Ricker and fitted params
preds_single_estimated, preds_single_estimated_log_abs = model.ricker_simulate(1, its, theta_hat,
                                            init=(initial_population_mean, 0),
                                            obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries, historic_mean, preds_single_estimated, its , test_index,
                             pars = 'estimated',
                             phi = "Estimated parameters",
                             var=historic_var)

# 2. 1. Forecast with ensemble of initial conditions.
preds_ensemble_estimated, preds_ensemble_estimated_log_abs = model.ricker_simulate(ensemble_size, its, theta_hat,
                                                 init=(initial_population_mean, ensemble_uncertainty),
                                                 obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_estimated, its , test_index,
                             pars = 'estimated',
                             phi = "Estimated parameters / Ensemble", var=historic_var)

# 3. Forecast with Ricker and known params
preds_single_perfect, preds_single_perfect_log_abs = model.ricker_simulate(1, its, theta_true,
                                            init=(initial_population_mean, 0),
                                            obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries, historic_mean, preds_single_perfect, its, test_index,
                             pars = 'perfect',
                             phi = "Perfect model knowledge", var=historic_var)

# 3. 1. Forecast with ensemble of initial conditions.
preds_ensemble_perfect, preds_ensemble_perfect_log_abs = model.ricker_simulate(ensemble_size, its, theta_true,
                                               init=(initial_population_mean, ensemble_uncertainty),
                                               obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_perfect, its, test_index,
                             pars = 'perfect',
                             phi = "Perfect model knowledge / Ensemble", var=historic_var)

#=======================#
# Evaluate forecasts    #
#=======================#

# 1.  Standard proficiency measure: RMSE

# Use performance of historic mean as forecast proficiency threshold.
fpt_hm = utils.rmse(timeseries[test_index:], historic_mean)

fed_estimated_params = utils.forecast_rmse(timeseries, preds_ensemble_estimated, test_index)
fed_perfect_model = utils.forecast_rmse(timeseries, preds_ensemble_perfect, test_index)

# 2. Forecast horizon as defined in Petchey 2015:
# When the mean of the forecast distribution falls below the forecast proficiency threshold.
# Example: Correlation in a moving window of size 3, threshold 0.5.

fpt_corr = 0.5
fcors_estimated_params = utils.rolling_corrs(timeseries, preds_ensemble_estimated, test_index)
fcors_perfect_model = utils.rolling_corrs(timeseries, preds_ensemble_perfect, test_index)

# 3. Lyapunov exponents

lyapunovs_estimated = np.mean(preds_ensemble_estimated_log_abs, axis=1)
lyapunovs_perfect = np.mean(preds_ensemble_perfect_log_abs, axis=1)

def lyapunov_efh(lyapunovs, Delta = 0.5, dell_0 = 0.01):
    return 1/lyapunovs*np.log(Delta/dell_0)


#==============#
# Plot results #
#==============#

vizualisations.forecast_error_distributions(fed_estimated_params, fpt_hm, 'estimated')
vizualisations.forecast_error_distributions(fed_perfect_model, fpt_hm, 'perfect')
vizualisations.forecast_corr_distributions(fcors_estimated_params, fpt_corr, 'both',
                                           mat2= fcors_perfect_model)




