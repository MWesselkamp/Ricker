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

def efh_mean(metric, profiencies, threshold, ps = False):
    """
    1. Function parameter: threshold.
    """
    profiencies_mean = profiencies.mean(axis=0)

    if metric == 'corr':
        efh = np.array([i < threshold for i in profiencies])
        pred_skills = np.argmax(profiencies < threshold, axis=1)
        #mean_pred_skill = min(np.arange(profiencies.shape[1])[profiencies_mean < threshold])
    elif metric == 'mse':
        efh = np.array([i > threshold for i in profiencies])
        pred_skills = np.argmax(profiencies > threshold, axis=1)
        # pred_skills = [min(np.arange(profiencies.shape[1])[efh[i,:]]) for i in range(profiencies.shape[0])]
        # mean_pred_skill = min(np.arange(profiencies.shape[1])[profiencies_mean > threshold])
    b = [np.sum(efh, axis=1) == 0]  # get the rows where the efh is never reached
    pred_skills[b] = profiencies.shape[1] # replace by maximum efh

    if ps:
        return pred_skills
    else:
        return efh, pred_skills#, mean_pred_skill

efh_mse, pred_skills  = efh_mean('mse', mse_rolling, 0.5)
fig = plt.figure()
plt.pcolor(efh_mse)
fig.show()

# Predict Forecast horizon under varying threshold for proficiency metrics.
threshold_seq = np.linspace(initial_uncertainty, 1, 20)
efhs_mse = np.array([efh_mean('mse', mse_rolling, t, ps=True) for t in threshold_seq])

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(threshold_seq, efhs_mse, color="lightblue")
plt.plot(threshold_seq, np.mean(efhs_mse, axis=1), color="darkblue")
ax.set_xlabel('MSE threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_mean/mse_threshold.png')


efh_corr, efh_corr_min = efh_mean('corr', corrs, 0.5)
efhs_corrs = np.array([efh_mean('corr', corrs, t, ps=True) for t in threshold_seq])

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(threshold_seq, efhs_corrs, color="lightblue")
plt.plot(threshold_seq, np.mean(efhs_corrs, axis=1), color="darkblue")
ax.set_xlabel('Correlation threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_mean/corr_threshold.png')


# For the correlation now consider the moving window as additional parameter.
mcorrs  = []
for wind in range(3,10):
    mcorrs.append(proficiency_metrics.rolling_corrs(ts_true, perfect_ensemble_d, window=wind))
efhs_mcorrs = np.array([efh_mean('corr', mcorrs[i], t, ps=True) for t in threshold_seq for i in range(len(mcorrs))])

efhs_mcorrs = efhs_mcorrs.reshape(20, 7, ensemble_size)
ehfs_mcorrs_m = np.mean(efhs_mcorrs, axis=2)

fig = plt.figure()
ax = fig.add_subplot()
for i in range(ensemble_size):
    plt.plot(threshold_seq, efhs_mcorrs[:,:,i], color="lightblue")
plt.plot(threshold_seq, ehfs_mcorrs_m, color="darkblue")
plt.plot(threshold_seq, np.mean(ehfs_mcorrs_m, axis=1), color="yellow")
ax.set_xlabel('Correlation threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_mean/corr_threshold_window.png')

# We require a second function parameter: The correlation strongly meanders around the threshold.
# For example: The EFH is the mean time after which the correlations falls below the threshold for at least three time steps in a row.
# Three days is super randomly?!
# Simply summarize this (following the definiton of Petchey: empirical confidence intervalls, or quantiles - looks very similar.)


# Quantile Horizon
def efh_quantile(metric, accepted_error, actual_error, timesteps, quantiles = (0.01, 0.99), ps = False):
    """
    1. Function parameter: What quantiles to use?
    2. What is the "expected error"
    """
    # Petcheys empirical Confidence Intervalls.
    def empCL(x, percent):
        ex = np.sort(x)[np.floor(percent / 100 * len(x)).astype(int)]
        return (ex)
    q_lower = [empCL(actual_error[:, i], quantiles[0]*100) for i in range(actual_error.shape[1])]
    q_mid = [empCL(actual_error[:, i], 50) for i in range(actual_error.shape[1])]
    q_upper = [empCL(actual_error[:, i], quantiles[1]*100) for i in range(actual_error.shape[1])]
    # Simply taking Quantiles
    error_metrics = ['mse', 'abs_diff']
    qu = np.quantile(actual_error, (quantiles[0], quantiles[1]), axis=0)
    efh = []
    for i in range(timesteps):
        if metric in error_metrics:
            e = not (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        elif metric == 'cor':
            e = (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        efh.append(e)
    if np.sum(efh) == 0:
        min_pred_skill = timesteps
    else:
        min_pred_skill = min(np.arange(len(efh))[efh])
    if ps:
        return min_pred_skill
    else:
        return efh, min_pred_skill


efh_corrs, efh_corrs_min  = efh_quantile('cor', 0.5, corrs, corrs.shape[1])
efh_corrs2, efh_corrs2_min = efh_quantile('cor', 0.5, corrs, corrs.shape[1], quantiles=(0.45, 0.55))
fig = plt.figure()
plt.plot(efh_corrs)
plt.plot(efh_corrs2)
fig.show()

# For varying threshold
def create_quantiles(n, max):
    u = 0.5 + np.linspace(0, max, n+1)
    l = 0.5 - np.linspace(0, max, n+1)
    r = np.array((l, u)).T
    return r[1:,]

qs = create_quantiles(20, max = 0.49)
efh_corrs_ps = np.array([efh_quantile('cor', j, corrs, corrs.shape[1], ps=True, quantiles=qs[q,:]) for j in threshold_seq for q in range(len(qs))])
efh_corrs_ps = efh_corrs_ps.reshape(20,20)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(threshold_seq, efh_corrs_ps, color="lightblue")
plt.plot(threshold_seq, np.mean(efh_corrs_ps, axis=1), color="darkblue")
ax.set_xlabel('Correlation threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_quantile/corr_threshold.png')


efh_abs_diff, efh_abs_diff_min = efh_quantile('abs_diff', initial_uncertainty, abs_diff, its)
fig = plt.figure()
plt.plot(efh_abs_diff)
fig.show()
# For varying threshold
efh_abs_diff_ps = np.array([efh_quantile('abs_diff', i, abs_diff, its, ps=True, quantiles=qs[q,:]) for i in threshold_seq for q in range(len(qs))])
efh_abs_diff_ps = efh_abs_diff_ps.reshape(20,20)
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(threshold_seq, efh_abs_diff_ps, color="lightblue")
plt.plot(threshold_seq, np.mean(efh_abs_diff_ps, axis=1), color="darkblue")
ax.set_xlabel('Absolute difference threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_quantile/absdiff_threshold.png')


efh_mse_rolling, efh_mse_rolling_min = efh_quantile('mse', initial_uncertainty, mse_rolling, its)
fig = plt.figure()
plt.plot(efh_mse_rolling)
fig.show()
# For varying threshold
efh_mse_ps  = np.array([efh_quantile('mse', i, mse_rolling, its, ps=True, quantiles=qs[q,:]) for i in threshold_seq for q in range(len(qs))])
efh_mse_ps = efh_mse_ps.reshape(20,20)
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(threshold_seq, efh_mse_ps, color="lightblue")
plt.plot(threshold_seq, np.mean(efh_mse_ps, axis=1), color="darkblue")
ax.set_xlabel('MSE threshold for acceptance')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(f'plots/baseline/efh_quantile/mse_threshold.png')


# t-test Statistics
t_stats, p_vals = proficiency_metrics.t_statistic(abs_diff, initial_uncertainty) # Where p-value is smaller than threshold.


#===============================#
# The Lyapunov forecast horizon #
#===============================#

lyapunovs = dynamics.lyapunovs(perfect_ensemble_derivative_d)
lyapunovs_stepwise = dynamics.lyapunovs(perfect_ensemble_derivative_d, stepwise=True)

l = 50
Delta_range = np.linspace(initial_uncertainty*1000, abs_diff.max(), l)
delta_range = np.linspace(initial_uncertainty, initial_uncertainty*100, l)

predicted_efh = np.array([dynamics.efh_lyapunov(lyapunovs, Delta, delta) for Delta in Delta_range for delta in delta_range])
predicted_efh = predicted_efh.reshape((l, l, ensemble_size))
predicted_efh_m = np.mean(predicted_efh, axis=2)

fig = plt.figure()
ax = fig.add_subplot()
for i in range(ensemble_size):
    plt.plot(Delta_range, predicted_efh[:,:,i], color="lightblue")
plt.plot(Delta_range, predicted_efh_m, color="darkblue")
plt.plot(Delta_range, np.mean(predicted_efh_m, axis=1), color="yellow")
ax.set_xlabel('Forecast proficiency threshold ($\Delta$)')
ax.set_ylabel('Predicted forecast horizon')
fig.savefig(r'plots/baseline/efh_lyapunov/delta_U.png')
fig.show()

predicted_efh = np.array([dynamics.efh_lyapunov(lyapunovs, Delta, delta) for delta in delta_range for Delta in Delta_range])
predicted_efh = predicted_efh.reshape((l, l, 50))
predicted_efh_m = np.mean(predicted_efh, axis=2)

fig = plt.figure()
ax = fig.add_subplot()
for i in range(ensemble_size):
    plt.plot(delta_range, predicted_efh[:,:,i], color="lightblue")
plt.plot(delta_range, predicted_efh_m, color="darkblue")
plt.plot(delta_range, np.mean(predicted_efh_m, axis=1), color="yellow")
ax.set_xlabel('Initial uncertainty ($\delta$)')
ax.set_ylabel('Predicted forecast horizon')
fig.savefig(r'plots/baseline/efh_lyapunov/delta_L.png')
fig.show()
#================================================================#
# Forecast horizon based on  Spring & Ilyina 2018 / Goddard 2013 #
#================================================================#

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