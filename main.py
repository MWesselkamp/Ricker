import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('module://backend_interagg')

import vizualisations
import utils
import model
import fit

import numpy as np

seed = 100
plot_results = False
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
ensemble_uncertainty = 1e-5

timeseries, timeseries_lyapunovs = model.ricker_simulate(1, its, theta_true,
                                             init = (initial_population_mean, initial_uncertainty),
                                             obs_error=False)

x_train = timeseries[:train_size]
x_test = timeseries[test_index:]

#=======================#
# Fit Ricker to x_train.#
#=======================#

## 1. Approach for data with purely deterministic model: Least squares.
lsq_solution, theta_hat = fit.lsq_fit(x_train, theta_true)

#=======================#
# Forecast time series  #
#=======================#

# 1. Forecast with historic mean
historic_mean, historic_var = model.historic_mean(x_test, x_train)

# 2. Forecast with Ricker and fitted params
preds_single_estimated, lyapunovs_single_estimated = model.ricker_simulate(1, its, theta_hat,
                                            init=(initial_population_mean, 0))

# 2. 1. Forecast with ensemble of initial conditions.
preds_ensemble_estimated, lyapunovs_ensemble_estimated = model.ricker_simulate(ensemble_size, its, theta_hat,
                                                 init=(initial_population_mean, ensemble_uncertainty))

# 3. Forecast with Ricker and known params
preds_single_perfect, lyapunovs_single_perfect = model.ricker_simulate(1, its, theta_true,
                                            init=(initial_population_mean, 0))

# 3. 1. Forecast with ensemble of initial conditions.
preds_ensemble_perfect, lyapunovs_ensemble_perfect = model.ricker_simulate(ensemble_size, its, theta_true,
                                               init=(initial_population_mean, ensemble_uncertainty))


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

# 3. EFH with Lyapunov exponents


#==============#
# Plot results #
#==============#

if plot_results:

    vizualisations.plot_forecast(timeseries, historic_mean, preds_single_estimated, its, test_index, pars = 'estimated', phi = "Estimated parameters", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_estimated, its , test_index, pars = 'estimated', phi = "Estimated parameters / Ensemble", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_single_perfect, its, test_index, pars = 'perfect', phi = "Perfect model knowledge", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_perfect, its, test_index, pars = 'perfect', phi = "Perfect model knowledge / Ensemble", var=historic_var)
    vizualisations.forecast_error_distributions(fed_estimated_params, fpt_hm, 'both', mat2 = fed_perfect_model)
    vizualisations.forecast_corr_distributions(fcors_estimated_params, fpt_corr, 'both', mat2= fcors_perfect_model)



#=============================#
# Varying the model parameter #
#=============================#

log_r_values = np.linspace(0.0, 2.0, 20)

true = list(zip(*[model.ricker_simulate(1, its, {'log_r':r, 'sigma':None, 'phi':10},
                                             init = (initial_population_mean, initial_uncertainty)) for r in log_r_values]))
predicted = list(zip(*[model.ricker_simulate(ensemble_size, its, {'log_r':r, 'sigma':None, 'phi':10},
                                             init = (initial_population_mean, ensemble_uncertainty)) for r in log_r_values]))

true_timeseries = np.array(true[0])
true_lyapunovs = np.array(true[1])
predicted_timeseries = np.array(predicted[0])
predicted_lyapunovs = np.array(predicted[1])

plot_results = True

vizualisations.plot_lyapunov_exponents(log_r_values, true_lyapunovs, predicted_lyapunovs)

def lyapunov_efh(lyapunovs, precision, dell0):
    return np.multiply(1/lyapunovs,np.log(precision/dell0))

precision = 0.8
predicted_efhs = lyapunov_efh(predicted_lyapunovs, precision, ensemble_uncertainty)

vizualisations.plot_lyapunov_efhs(log_r_values, predicted_efhs)
vizualisations.plot_lyapunov_efhs(log_r_values, predicted_efhs, log = True)

absolute_differences = abs(np.subtract(true_timeseries.reshape((20,1,100)), predicted_timeseries))
historic_mean, historic_var = model.historic_mean(x_test, x_train, length=100)
differences_historic_mean = abs(np.subtract(true_timeseries, historic_mean))


if plot_results:
    for i in range(20):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(np.arange(100), np.mean(absolute_differences[i,:,:], axis=0), color="black", label="predicted")
        plt.plot(np.arange(100), differences_historic_mean[i, :], color="red", label="historic mean")
        ax.set_ylabel("Absolute difference to truth")
        ax.set_xlabel("Time step")
        fig.show()

#=============================================#
# Predictability skill based on Séférian 2013 #
#=============================================#
# And: http://www.bom.gov.au/wmo/lrfvs/msss.shtml

# Test the RMSEE against the standard deviation of observations, i.e. inital_uncertainty.

def mse_f(predicted, true, n):
    return (1/n)*np.sum((predicted-true)**2)

def mse_cj(true, n):
    return (1/n)*np.sum((true - np.mean(true))**2)

mses = []
for i in range(20):
    mse1 = mse_f(predicted_timeseries[i,:,:], true_timeseries[i,:], its)
    mse2 = mse_cj(true_timeseries[i,:], its)
    rat = 1 - (mse1/mse2)
    mses.append(rat)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(log_r_values, mses)
ax.set_ylabel("Mean squared skill score")
ax.set_xlabel("Log r value")
fig.show()

#==================================================#
# Calculate the F-statistic at every point in time #
#==================================================#

# # i.e. the ratio of RMSEE variance against observation variance.


## get a p-value for this difference and save it


def horizon(condition):

    """
    Calculate the horizon for a given condition: e.g. p-value < 0.05
    :param condition:
    :return: x.array Dataset with information about forecast, units...
    """
