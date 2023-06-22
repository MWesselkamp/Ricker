import dynamics
import simulations
import utils
import visualisations
import forecast_ensemble
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)

#==========================================#
# The Ricker model for population dynamics #
#==========================================#
sims = models.Simulator(model_type="single-species",
                        environment="exogeneous",
                        growth_rate=.05, # non chaotic
                        timesteps=365,
                        ensemble_size=5,
                        initial_size=1)

xsim = sims.simulate(sigma= 0.00,phi= 0.00,initial_uncertainty=0.05)['ts_obs']
vizualisations.baseplot(xsim, transpose=True)

obs = models.Simulator(model_type="multi-species",
                       environment="exogeneous",
                       growth_rate=.05,
                       timesteps=365,
                       ensemble_size=1,
                       initial_size=(1, 1))
xobs1 = obs.simulate(sigma= 0.00,phi= 0.00,initial_uncertainty= 0.05)['ts_obs'][:,:,0]
xobs2 = obs.simulate(sigma= 0.00,phi= 0.00,initial_uncertainty= 0.05)['ts_obs'][:,:,1]

vizualisations.baseplot(xsim, x2=xobs1, x3=xobs2, transpose=True)



#=============================#
# Verification: Perfect model #
#=============================#
perfect_ensemble = forecast_ensemble.PerfectEnsemble(ensemble_predictions=xsim,
                                                     reference="rolling_climatology")
perfect_ensemble.verification_settings(metric = "rolling_corrs",
                                       evaluation_style="bootstrap")
perfect_ensemble.accuracy()



if perfect_ensemble.meta['evaluation_style'] == "single":
    control = xsim[perfect_ensemble.meta['other']['ensemble_index']]
    vizualisations.baseplot(xsim, control, transpose=True)
    vizualisations.baseplot(perfect_ensemble.accuracy_model, perfect_ensemble.accuracy_reference,
                            transpose=True, ylab=perfect_ensemble.meta['metric'])
elif perfect_ensemble.meta['evaluation_style'] == "bootstrap":
    vizualisations.baseplot(xsim, np.array(perfect_ensemble.reference_simulation).squeeze(), transpose=True)
    vizualisations.baseplot(np.mean(perfect_ensemble.accuracy_model, axis=0), np.mean(perfect_ensemble.accuracy_reference,axis=0),
                            transpose=True, ylab=perfect_ensemble.meta['metric'])


forecast_skill = perfect_ensemble.skill()
x = np.mean(forecast_skill, axis=0)
vizualisations.baseplot(x, transpose=True,
                        ylab=f"Relative {perfect_ensemble.meta['metric']}")

#============================================================#
# What do you want do want to validate the forecast against? #
#============================================================#

# The reference: Here Observations (simulated but who cares)
# As such, is this hindcasting?

obs = models.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
obs.hyper_parameters(simulated_years=10,
                           ensemble_size=1,
                           initial_size=(1, 1))
xobs = obs.simulate()[:,:,0]
dell_0 = abs(xsim[:,0]-xobs[:,0])

imperfect_ensemble = forecast_ensemble.ImperfectEnsemble(ensemble_predictions=xsim,
                                                           observations=xobs,
                                                            reference="rolling_climatology")
imperfect_ensemble.verification_settings(metric = "rolling_rmse",
                                       evaluation_style="single")
imperfect_ensemble.accuracy()
vizualisations.baseplot(imperfect_ensemble.accuracy_model,imperfect_ensemble.accuracy_reference,
                        transpose=True, ylab = imperfect_ensemble.meta['metric'])

forecast_skill = imperfect_ensemble.skill()
vizualisations.baseplot(forecast_skill, transpose=True,
                        ylab=f"relative {imperfect_ensemble.meta['metric']}")

#===================#
# Now with forecast #
#===================#

sims.forecast(years = 5, observations=xobs)
xpred = sims.forecast_simulation['ts']
vizualisations.baseplot(xpred, transpose=True)

prediction_ensemble = forecast_ensemble.ForecastEnsemble(ensemble_predictions=xpred,
                                                           observations=xobs,
                                                            reference="climatology")
prediction_ensemble.verification_settings(metric = "rolling_rmse",
                                       evaluation_style="single")

prediction_ensemble.forecast_accuracy()

ref_mean = prediction_ensemble.reference_mean
ref_var = prediction_ensemble.reference_var
vizualisations.baseplot(xpred, ref_mean, transpose=True,
                        ylab="Population size")

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.transpose(ref_mean), alpha=0.8, color="red")
ax.fill_between(np.arange(ref_mean.shape[1]),np.transpose(ref_mean+2*ref_var).squeeze(), np.transpose(ref_mean-2*ref_var).squeeze(),
                alpha=0.3, color="red")
ax.plot(np.transpose(xpred), alpha=0.8, color="blue")
#ax.set_xlabel(xlab, size=14)
#ax.set_ylabel(ylab, size=14)
fig.show()

vizualisations.baseplot(prediction_ensemble.forecast_accuracy, transpose=True,
                        ylab=f"{prediction_ensemble.meta['metric']}")


#========#
# horizon#
#========#

# Probability of exceeding threshold
import horizons
exp_fsh, var_fsh, fhs = horizons.forecastskill_mean(prediction_ensemble.forecast_accuracy, threshold=0.0006)
vizualisations.baseplot(exp_fsh, exp_fsh+var_fsh, exp_fsh-var_fsh)

exp_fsh = horizons.mean_forecastskill(prediction_ensemble.forecast_accuracy, threshold=0.0006)
vizualisations.baseplot(exp_fsh[0])


def raw_CNR(obs, pred, squared = False):
    """
    CNR - contrast to noise ratio: mean(condition-baseline) / std(baseline)
    This is basically the same as the square-error-based SNR?
    Transfered, we have the model as the baseline and the mean as condition.
    tsnr increases with sample size (see sd).
    """
    signal = np.mean((pred - np.mean(obs))) # returns a scalar
    noise = np.std(obs)
    if squared:
        return signal**2/noise**2, signal**2, noise**2
    else:
        return signal/noise, signal, noise

cnrs = []
for i in range(1,xpred.shape[1]):
    cnrs.append(raw_CNR(xpred[:,:i], xobs[:,i]))
cnrs = np.array(cnrs)
vizualisations.baseplot(cnrs)

import matplotlib.pyplot as plt
# An "interface" to matplotlib.axes.Axes.hist() method
l = np.linspace(0.01,0.99,40)

qs = np.quantile(xobs.flatten(), l)
hist_clim, bin_edges = np.histogram(xobs, bins = qs, range=(xobs.min(), xobs.max()), density=True)
probs = hist_clim*np.diff(bin_edges)

xpreds = xpred[:,:150]
qs = np.quantile(xobs.flatten(), l)
hist_clim_pred, bin_edges_pred = np.histogram(xpreds.flatten(), bins = qs, range=(xpreds.min(), xpreds.max()), density=True)
probs_pred = hist_clim_pred*np.diff(bin_edges_pred)

fig = plt.figure()
plt.bar(bin_edges[:-1],hist_clim/hist_clim.sum(), width=0.0002)
plt.bar(bin_edges_pred[:-1],hist_clim_pred/hist_clim_pred.sum(), width=0.0002, alpha = 0.4, color="red")
fig.show()


#===============#
# t-test horizon#
#===============#

from scipy.stats import ttest_ind

tstats = []
pvalues = []
for i in range(1, xpred.shape[1]):
    ttest_results = ttest_ind(xobs.flatten(), xpred[:,:i].flatten(), equal_var=False)
    tstats.append(ttest_results.statistic)
    pvalues.append(ttest_results.pvalue)

fig = plt.figure()
plt.plot(tstats, color="blue")
plt.plot(pvalues, color= "red")
fig.show()

#===============================#
# The Lyapunov forecast horizon #
#===============================#

lyapunovs_stepwise = dynamics.lyapunovs(xsim_derivative, stepwise=True)
vizualisations.baseplot(lyapunovs_stepwise)

l = 30
initial_uncertainty_estimate = np.mean(abs((xsim[:,0]-xobs[:,0])))
min_D = utils.min_Delta(initial_uncertainty_estimate)

Delta_range = np.linspace(min_D, abs_diff.max(), l)
delta_range = np.linspace(initial_uncertainty_estimate, initial_uncertainty_estimate*1000, l)

vizualisations.ln_deltaRatio(delta_range, Delta_range, l)


predicted_efh = np.array([dynamics.efh_lyapunov(lyapunovs, Delta, delta) for Delta in Delta_range for delta in delta_range])
predicted_efh = predicted_efh.reshape((l, l, ensemble_size))
predicted_efh_m = np.mean(predicted_efh, axis=2)
vizualisations.delta_U(Delta_range, predicted_efh, predicted_efh_m, ensemble_size)

predicted_efh = np.array([dynamics.efh_lyapunov(lyapunovs, Delta, delta) for delta in delta_range for Delta in Delta_range])
predicted_efh = predicted_efh.reshape((l, l, 50))
predicted_efh_m = np.mean(predicted_efh, axis=2)
vizualisations.delta_L(delta_range, predicted_efh, predicted_efh_m, ensemble_size)


potential_fh = np.max(predicted_efh_m, axis=1)
potential_fh_roc = abs(potential_fh[1:]-potential_fh[:-1])
print("Potential forecast horizon at varying initial uncertainties:", potential_fh)
print("Rates of change potential FH: ", potential_fh_roc)
print("Potential forecast horizon at varying initial uncertainties for a Delta of:", Delta_range[np.argmax(predicted_efh_m, axis=1)[0]])

vizualisations.potential_fh_roc(potential_fh_roc, delta_range)

# For a fixed Horizon that we want to reach, e.g 20
Tp = 20
tp_Delta = utils.fixed_Tp_Delta(lyapunovs, Tp, delta_range)
tp_delta = utils.fixed_Tp_delta(lyapunovs, Tp, Delta_range)
vizualisations.fixed_Tp_delta(delta_range, Delta_range, tp_Delta, tp_delta)

#======================================#
# Lyapunov EFH under varying r values  #
#======================================#

len_r_values = 30
r_values = np.linspace(0.2, 8, len_r_values)
l = 50
Delta_range = np.linspace(initial_uncertainty_estimate, 2, l)
delta = initial_uncertainty_estimate
fix_lin1 = np.linspace(0.1, 1, l)
fix_lin2 = np.linspace(1, 2.5, l)
# Initalize model
ricker = modelclass.Ricker(initial_size, initial_uncertainty)
predicted_efhs_ms = []
predicted_efhs_fix1_ms = []
predicted_efhs_fix2_ms = []
lyapunovs = []

for r in r_values:

    theta = {'r':r, 'sigma':0.3}
    ricker = modelclass.Ricker(initial_size, initial_uncertainty)
    ricker.set_parameters(theta = theta)
    #ricker.print_parameters()
    simulator = modelclass.Simulation(ricker, iterations=its) # Create a simulator object
    # To simulate the baseline ensemble
    simulator.sources_of_uncertainty(parameters=False,
                                    initial = True,
                                    observation = False,
                                    stoch = False)
    perfect_ensemble_d, perfect_ensemble_derivative_d = simulator.simulate(ensemble_size)
    lyas = dynamics.lyapunovs(perfect_ensemble_derivative_d)
    lyapunovs.append(lyas)

    predicted_efh = np.array([dynamics.efh_lyapunov(lyas, Delta, delta) for Delta in Delta_range])
    predicted_efh_fix1 = np.array([dynamics.efh_lyapunov(lyas, Delta=None, delta=None, fix = fix) for fix in fix_lin1])
    predicted_efh_fix2 = np.array([dynamics.efh_lyapunov(lyas, Delta=None, delta=None, fix=fix) for fix in fix_lin2])

    predicted_efhs_ms.append(np.mean(predicted_efh, axis=1))
    predicted_efhs_fix1_ms.append(np.mean(predicted_efh_fix1, axis=1))
    predicted_efhs_fix2_ms.append(np.mean(predicted_efh_fix2, axis=1))

predicted_efhs_ms = np.array(predicted_efhs_ms)
predicted_efhs_fix1_ms = np.array(predicted_efhs_fix1_ms)
predicted_efhs_fix2_ms = np.array(predicted_efhs_fix2_ms)
lyapunovs = np.array(lyapunovs)

vizualisations.lyapunov_time(predicted_efhs_ms, r_values)
vizualisations.lyapunov_time_modifier_effect(r_values, predicted_efhs_fix1_ms, predicted_efhs_fix2_ms)
vizualisations.lyapunovs_along_r(r_values, lyapunovs)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.transpose(perfect_ensemble_derivative_d), color="lightgrey")
fig.show()

#================================================================#
# Forecast horizon Spring & Ilyina 2018 / Goddard 2013 #
#================================================================#

# Required: Bootstrap perfect ensemble function:
# Required: Skill metric (default: Pearsons r.)

#for i in range(ensemble_size):
#    leftout = perfect_ensemble_d[i,:] # assign the leftout trajectory
#    rest = np.delete(perfect_ensemble_d, i, 0) # assign the rest

#=================================================#
# Forecast horizon  Séférian et al 2018 #
#=================================================#
