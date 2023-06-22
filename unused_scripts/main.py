import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('module://backend_interagg')

import visualisations
import utils
from code_graveyard import model
import fit
import dynamics
import numpy as np

seed = 100
plot_results = False
# initialize random number generator

#===========================#
# Simulate true time series #
#===========================#

theta_true = {'r':2.9, 'sigma':None}
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
historic_mean, historic_var = utils.historic_mean(x_test, x_train)

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

#==============#
# Plot results #
#==============#

if plot_results:

    vizualisations.plot_forecast(timeseries, historic_mean, preds_single_estimated, its, test_index, pars = 'estimated', phi = "Estimated parameters", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_estimated, its , test_index, pars = 'estimated', phi = "Estimated parameters / Ensemble", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_single_perfect, its, test_index, pars = 'perfect', phi = "Perfect model knowledge", var=historic_var)
    vizualisations.plot_forecast(timeseries, historic_mean, preds_ensemble_perfect, its, test_index, pars = 'perfect', phi = "Perfect model knowledge / Ensemble", var=historic_var)

#============================#
# Forecast proficiency: RMSE #
#============================#

# Use performance of historic mean as forecast proficiency threshold.
historic_mean, historic_var = utils.historic_mean(x_test, x_train, length=its)
#fpt_hm = utils.rmse(timeseries, historic_mean)

#fed_estimated_params = utils.forecast_rmse(timeseries, preds_ensemble_estimated, test_index=0)
#fed_perfect_model = utils.forecast_rmse(timeseries, preds_ensemble_perfect, test_index = 0)
#vizualisations.FP_rmse(fed_perfect_model, fpt_hm, 'both', mat2=fed_perfect_model)

#===================================#
# Forecast proficiency: Correlation #
#===================================#
# When the mean of the forecast distribution falls below the forecast proficiency threshold.
# Example: Correlation in a moving window of size 3, threshold 0.5.

#fpt_corr = 0.5
#fcors_estimated_params = utils.rolling_corrs(timeseries, preds_ensemble_estimated, test_index=0)
#fcors_perfect_model = utils.rolling_corrs(timeseries, preds_ensemble_perfect, test_index=0)
#vizualisations.FP_correlation(fcors_perfect_model, fpt_corr, 'both', mat2= fcors_perfect_model)

#=============================================#
# Forecast proficiency: Absolute differences  #
#=============================================#

#absolute_differences = np.transpose(abs(np.subtract(timeseries, preds_ensemble_perfect)))
#absolute_differences_mean = np.mean(absolute_differences, axis=1)

#vizualisations.FP_absdifferences(absolute_differences, absolute_differences_mean, its)
#vizualisations.FP_absdifferences(absolute_differences, absolute_differences_mean, its, log=True)

# threshold?

#==================================#
# Lyapunov EFH for known r values  #
#==================================#

lyapunovs = np.array([np.mean(lyapunovs_ensemble_perfect[:,1:i], axis=1) for i in range(lyapunovs_ensemble_perfect.shape[1])])
lyapunovs_over_time = np.mean(lyapunovs, axis=1)
lyapunovs_full = lyapunovs[-1,:]

# Absolute Differences as forecast profieciency
# Maximum difference:
Delta_max = abs_diff.max()
Delta_range = np.linspace(ensemble_uncertainty, Delta_max, 40)

predicted_efh = np.array([dynamics.efh_lyapunov(lyapunovs_full, Delta, ensemble_uncertainty) for Delta in Delta_range])


# RMS as Forecast proficiency


#======================================#
# Lyapunov EFH under varying r values  #
#======================================#

len_r_values = 30
r_values = np.linspace(0.001, 4.0, len_r_values)

true = list(zip(*[model.ricker_simulate(1, its, {'r':r, 'sigma':None},
                                        init = (initial_population_mean, initial_uncertainty)) for r in r_values]))
true_timeseries = np.array(true[0])
true_lyapunovs = np.mean(np.array(true[1]).reshape(len_r_values, its), axis=1) # only last calculated Lyapunov exponent


predicted = list(zip(*[model.ricker_simulate(ensemble_size, its, {'r':r, 'sigma':None},
                                             init = (initial_population_mean, ensemble_uncertainty)) for r in r_values]))

predicted_timeseries = np.array(predicted[0])
predicted_lyapunovs = np.mean(np.array(predicted[1]), axis=2)

plot_results = True
# plot Lyapunov exponent at end of time series.
vizualisations.plot_lyapunov_exponents(r_values, true_lyapunovs, predicted_lyapunovs)

precision_threshold = 0.8 # under varying precision?
predicted_efhs = dynamics.efh_lyapunov(predicted_lyapunovs, precision_threshold, ensemble_uncertainty)

vizualisations.plot_lyapunov_efhs(r_values, predicted_efhs)
vizualisations.plot_lyapunov_efhs(r_values, predicted_efhs, log = True)

#=================================================#
# Absolute differences of Forecasts to reference  #
#=================================================#

# Currently: Take observations as reference and mean and Ricker as forecasts.
# USE historic mean as REFERENCE instead of forecast!
absolute_differences = abs(np.subtract(true_timeseries.reshape((len_r_values,1,100)), predicted_timeseries))
historic_mean, historic_var = model.historic_mean(x_test, x_train, length=100)
differences_historic_mean = abs(np.subtract(true_timeseries, historic_mean))

plot_results=False

if plot_results:
    for i in range(len_r_values):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(np.arange(100), np.mean(absolute_differences[i,:,:], axis=0), color="black", label="predicted")
        plt.plot(np.arange(100), differences_historic_mean[i, :], color="red", label="historic mean")
        ax.set_ylabel("Absolute difference to truth")
        ax.set_xlabel("Time step")
        fig.show()

# Forecast horizon with absolute difference as Delta over time.


#=============================================#
# Predictability skill based on Séférian 2013 #
#=============================================#
# And: http://www.bom.gov.au/wmo/lrfvs/msss.shtml

# Only possible if OBSERVATIONS are present!
# Test the RMSEE against the standard deviation of observations, i.e. inital_uncertainty.

def mse_f(predicted, true, n):
    return (1/n)*np.sum((predicted-true)**2)

def mse_cj(true, n):
    return (1/n)*np.sum((true - np.mean(true))**2)

mses = []
for i in range(len_r_values):
    mse1 = mse_f(predicted_timeseries[i,:,:], true_timeseries[i,:], its)
    mse2 = mse_cj(true_timeseries[i,:], its)
    rat = 1 - (mse1/mse2)
    mses.append(rat)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(r_values, mses)
ax.set_ylabel("Mean squared skill score")
ax.set_xlabel("r value")
fig.show()

#==================================================#
# Calculate the F-statistic at every point in time #
#==================================================#

# # i.e. the ratio of RMSEE variance against observation variance.
# # OR the forecast variance against the historic mean variance.
# degrees of freedom?!


## get a p-value for this difference and save it


def horizon(condition):

    """
    Calculate the horizon for a given condition: e.g. p-value < 0.05
    :param condition:
    :return: x.array Dataset with information about forecast, units...
    """
