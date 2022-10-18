import dynamics
import simulations
import utils
import vizualisations
import forecast_ensemble
import numpy as np
import matplotlib.pyplot as plt

#==========================================#
# The Ricker model for population dynamics #
#==========================================#
sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
sims.hyper_parameters(simulated_years=1,
                           ensemble_size=30,
                           initial_size=950)
xsim = sims.simulate()
mod = sims.ricker
xsim_derivative = mod.derive(xsim)

perfect_ensemble = forecast_ensemble.PerfectEnsemble(ensemble_predictions=xsim,
                                                     reference="rolling_historic_mean")
perfect_ensemble.verification_settings(metric = "rolling_rsquared",
                                       evaluation_style="bootstrap")
perfect_ensemble.accuracy()

if perfect_ensemble.meta['evaluation_style'] == "single":
    control = xsim[perfect_ensemble.meta['other']['ensemble_index']]
    vizualisations.baseplot(xsim, control, transpose=True)
    vizualisations.baseplot(perfect_ensemble.accuracy_model, perfect_ensemble.accuracy_reference, transpose=True)
elif perfect_ensemble.meta['evaluation_style'] == "bootstrap":
    vizualisations.baseplot(np.mean(perfect_ensemble.accuracy_model, axis=0), np.mean(perfect_ensemble.accuracy_reference,axis=0), transpose=True)

forecast_skill = perfect_ensemble.skill()
x = np.mean(forecast_skill, axis=0)
vizualisations.baseplot(x, transpose=True)

#============================================================#
# What do you want do want to validate the forecast against? #
#============================================================#

# The reference: Here Observations (simulated but who cares)
# As such, is this hindcasting?

obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
obs.hyper_parameters(simulated_years=1,
                           ensemble_size=1,
                           initial_size=(950, 950))
xobs = obs.simulate()[:,:,0]
dell_0 = abs(xsim[:,0]-xobs[:,0])

prediction_ensemble = forecast_ensemble.PredictionEnsemble(ensemble_predictions=xsim,
                                                           observations=xobs,
                                                            reference="rolling_historic_mean")
prediction_ensemble.verification_settings(metric = "rolling_rsquared",
                                       evaluation_style="single")
prediction_ensemble.accuracy()
vizualisations.baseplot(prediction_ensemble.accuracy_model,prediction_ensemble.accuracy_reference, transpose=True)

forecast_skill = prediction_ensemble.skill()
vizualisations.baseplot(forecast_skill, transpose=True,
                        ylab="rolling_rsquared")

#===================#
# Now with forecast #
#===================#

sims.forecast(years = 2, observations=xobs)
xpred = sims.forecast_simulation['ts']
vizualisations.baseplot(xpred, transpose=True)

prediction_ensemble = forecast_ensemble.PredictionEnsemble(ensemble_predictions=xpred,
                                                           observations=xobs,
                                                            reference="persistance")
prediction_ensemble.verification_settings(metric = "rolling_rsquared",
                                       evaluation_style="single")

prediction_ensemble.accuracy()
forecast_skill = prediction_ensemble.skill()

v_ref = prediction_ensemble.reference_simulation
vizualisations.baseplot(xpred, v_ref, transpose=True,
                        ylab="Population size")
vizualisations.baseplot(forecast_skill, transpose=True,
                        ylab="Skill: rolling_rsquared")


expectation_fsh, dispersion_fsh = prediction_ensemble.horizon(fh_type = "mean_forecastskill", threshold=1)
# rather: Probability of exceeding threshold
expectation_fsh, dispersion_fsh = prediction_ensemble.horizon(fh_type="forecastskill_mean", threshold=1)

vizualisations.baseplot(expectation_fsh, expectation_fsh+dispersion_fsh, expectation_fsh+dispersion_fsh)


#===============#
# t-test horizon#
#===============#
t_stats, p_vals = proficiency_metrics.t_statistic(abs_diff, initial_uncertainty) # Where p-value is smaller than threshold.

#===============================#
# The Lyapunov forecast horizon #
#===============================#

lyapunovs = dynamics.lyapunovs(perfect_ensemble_derivative_d)
lyapunovs_stepwise = dynamics.lyapunovs(perfect_ensemble_derivative_d, stepwise=True)

l = 30
initial_uncertainty_estimate = np.mean(abs((perfect_ensemble_d[:,0]-ts_true[0])))

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
