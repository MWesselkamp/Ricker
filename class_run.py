import dynamics
import modelclass
import utils
import proficiency_metrics
import vizualisations
import numpy as np
import matplotlib.pyplot as plt

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
# The horizon with Quantiles    #
#===============================#

# Choose a forecast proficiency metric
# 1. Absolute difference to truth.
abs_diff, abs_diff_mean = proficiency_metrics.absolute_difference(ts_true, perfect_ensemble_d, mean = True)
vizualisations.FP_absdifferences(abs_diff, abs_diff_mean, its)

# 2. Rolling MSE
# Metric Parameter: Moving window
mse = proficiency_metrics.mean_squared_error(ts_true, perfect_ensemble_d)
mse_rolling = proficiency_metrics.mean_squared_error_rolling(ts_true, perfect_ensemble_d)

# 3. Rolling Correlation
# Metric Parameter: Moving window of size 3 (Petchey.)
corrs = proficiency_metrics.rolling_corrs(ts_true, perfect_ensemble_d, window=3)
#corrr = np.transpose(np.array(corr))
fig = plt.figure()
plt.plot(np.transpose(corrs))
fig.show()

def efh_mean(profiencies, threshold):
    """
    1. Function parameter: threshold.
    """

    def empCL(x, percent):
        ex = np.sort(x)[np.floor(percent / 100 * len(x)).astype(int)]
        return (ex)

    q_lower = [empCL(profiencies[:, i], 45) for i in range(profiencies.shape[1])]
    q_mid = [empCL(profiencies[:, i], 50) for i in range(profiencies.shape[1])]
    q_upper = [empCL(profiencies[:, i], 55) for i in range(profiencies.shape[1])]

    efh = np.array([i < threshold for i in profiencies])
    corrs_mean = corrs.mean(axis=0)
    min_pred_skill = min(np.arange(profiencies.shape[1])[corrs_mean < threshold])

    return efh, min_pred_skill

efh_corr, efh_corr_min = efh_mean(corrs, 0.5)
fig = plt.figure()
plt.pcolor(efh_corr)
fig.show()
# We require a second function parameter: The correlation seems to meander around the threshold.
# So for example: The EFH is the mean time after which the correlations falls below the threshold for at least three time steps in a row.
# This is super randomly?!
# So simply summarize this (following the definiton of Petchey:
# empirical confidence intervalls, or use quantiles - looks very similar.)

print("Mean EFH with rolling correlation: ", efh_corr_min)

# Quantile Horizon
def efh_quantile(metric, accepted_error, actual_error, timesteps, quantiles = (0.01, 0.99)):
    """
    1. Function parameter: What quantiles to use?
    2. What is the "expected error" (or the one we accept)? Depends on the metric we choose!
                                                            0.5 for correlation.
                                                            Currently initial uncertainty for MSE and absDiff
    """
    error_metrics = ['mse', 'abs_diff']
    qu = np.quantile(actual_error, (quantiles[0], quantiles[1]), axis=0)
    efh = []
    for i in range(timesteps):
        if metric in error_metrics:
            e = not (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        elif metric == 'cor':
            e = (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        efh.append(e)
    min_pred_skill = min(np.arange(len(efh))[efh])
    return efh, min_pred_skill


efh_corrs, efh_corrs_min  = efh_quantile('cor', 0.5, corrs, corrs.shape[1])
efh_corrs2, efh_corrs2_min = efh_quantile('cor', 0.5, corrs, corrs.shape[1], quantiles=(0.45, 0.55))
fig = plt.figure()
plt.plot(efh_corrs)
plt.plot(efh_corrs2)
fig.show()
print("Quantile EFH1 with correlation: ", efh_corrs_min)
print("Quantile EFH2 with correlation: ", efh_corrs2_min)


efh_abs_diff, efh_abs_diff_min = efh_quantile('abs_diff', initial_uncertainty, abs_diff, its)
fig = plt.figure()
plt.plot(efh_abs_diff)
fig.show()
print("Quantile EFH with absolute differences: ", efh_abs_diff_min)

efh_mse_rolling, efh_mse_rolling_min = efh_quantile('mse', initial_uncertainty, mse_rolling, its)
fig = plt.figure()
plt.plot(efh_mse_rolling)
fig.show()
print("Quantile EFH with Rolling MSE: ", efh_mse_rolling_min)

# t-test Statistics
t_stats, p_vals = proficiency_metrics.t_statistic(abs_diff, initial_uncertainty) # Where p-value is smaller than threshold.


#===============================#
# The Lyapunov forecast horizon #
#===============================#

lyapunovs = dynamics.lyapunovs(perfect_ensemble_derivative_d)
lyapunovs_stepwise = dynamics.lyapunovs(perfect_ensemble_derivative_d, stepwise=True)

Delta_range = np.linspace(initial_uncertainty, abs_diff.max()*2, 40)

predicted_efh = np.array([dynamics.lyapunov_efh(lyapunovs, Delta, initial_uncertainty) for Delta in Delta_range])
vizualisations.plot_LE_efh_along_Delta(Delta_range, predicted_efh)


#=================================================#
# Forecast horizon based on  Spring & Ilyina 2018 #
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