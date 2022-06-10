import matplotlib

import vizualisations

matplotlib.use('module://backend_interagg')
import model
from vizualisations import plot_posterior
from vizualisations import plot_trajectories
import math
import numpy as np
import scipy.optimize as optim

theta_true = {'log_r':2.8, 'sigma':0.3, 'phi':10}
samples = 1
its = 50 # Petchey 2015
init = (1, 0)

timeseries_array_obs, timeseries_array_true = model.ricker_simulate(samples, its, theta_true, init = init,
                                                                    obs_error=False, stoch=False)

x_train = timeseries_array_obs[0][:30]
x_test = timeseries_array_obs[0][31:]

#=======================#
# Fit Ricker to x_train.#
#=======================#

## 1. Approach for data with purely deterministic model: Least squares.

def ricker(N, log_r, phi):
    return np.exp(log_r+np.log(N)-N)*phi

# function to minimize
def fun(pars, x, y):
    res = ricker(x, pars[0], pars[1]) - y
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


#=======================#
# Forecast time series  #
#=======================#

# 1. Forecast with historic mean
historic_mean = np.full((x_test.shape[0]), np.mean(x_train), dtype=np.float)
historic_var = np.full((x_test.shape[0]), np.std(x_train), dtype=np.float)
print(historic_mean, historic_var)

# 2. Forecast with Ricker and fitted params
samples = 1
its=50
theta = {'log_r':log_r, 'sigma':None, 'phi':phi}
init = (1, 0)

ricker_preds1, timeseries_array_true = model.ricker_simulate(samples, its, theta, init=init, obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries_array_obs, historic_mean, ricker_preds1, its , phi = "Estimated parameters", var=historic_var)

# 2. 1. Forecast with ensemble of initial conditions.
samples = 10
its=50
theta = {'log_r':log_r, 'sigma':None, 'phi':phi}
init = (1, 0.01)

ricker_preds1, timeseries_array_true = model.ricker_simulate(samples, its, theta, init=init, obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries_array_obs, historic_mean, ricker_preds1, its , phi = "Estimated parameters", var=historic_var)

# 3. Forecast with Ricker and known params
samples = 1
its=50
theta = theta_true
init = (1, 0)

ricker_preds2, timeseries_array_true = model.ricker_simulate(samples, its, theta, init=init, obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries_array_obs, historic_mean, ricker_preds2, its, phi = "Perfect model knowledge", var=historic_var)

# 3. 1. Forecast with ensemble of initial conditions.

samples = 1
its=50
theta = theta_true
init = (1, 0.01)

ricker_preds2, timeseries_array_true = model.ricker_simulate(samples, its, theta, init=init, obs_error=False, stoch=False)
vizualisations.plot_forecast(timeseries_array_obs, historic_mean, ricker_preds2, its, phi = "Perfect model knowledge", var=historic_var)

#=======================#
# Evaluate forecasts    #
#=======================#

# 1.  Standard proficiency measure: RMSE
def rmse(y, y_pred):
    return math.sqrt(np.square(np.subtract(y,y_pred)).mean())

# Use performance of historic mean as forecast proficiency threshold.
fpt = rmse(timeseries_array_obs[0][31:], historic_mean)
print('Root mean square error:', fpt)

errors = []
for i in range(ricker_preds1.shape[0]):
    errors.append(rmse(timeseries_array_obs[0][31:], ricker_preds1[i,31:]))
print('Root mean square error:',errors)

# Forecast horizon as defined in Petchey 2015:
# When the mean of the forecast distribution falls below the forecast proficiency threshold.



# 2. Lyapunov exponent

