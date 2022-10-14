import matplotlib.pyplot as plt
import numpy as np

def baseplot(x1, x2=None, x3 = None, transpose=False, xlab=None, ylab=None):
    if transpose:
        x1 = np.transpose(x1)
        if not x2 is None:
            x2 = np.transpose(x2)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x1, alpha=0.3, color="blue")
    if not x2 is None:
        ax.plot(x2, alpha=0.3, color="red")
    if not x3 is None:
        ax.plot(x3, alpha=0.3, color="green")
    ax.set_xlabel(xlab, size=14)
    ax.set_ylabel(ylab, size=14)
    fig.show()

def plot_growth(real_dynamics):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(real_dynamics)
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
    fig.show()
    fig.savefig('plots/real_dynamics.png')

def plot_posterior(postsamples):
    """
    Write a function, that displays posterior distributions.
     Save plot to plots folder.
     Move it to a visualizations script.
     :param ps:
     :return:
     """

    fig = plt.figure()
    ax = fig.add_subplot()

    h = np.transpose(postsamples.numpy())
    plt.hist(h, bins=30)

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Sample frequency')
    fig.show()
    fig.savefig('plots/posterior.png')

def plot_trajectories(trajectories, its, true = None):

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(np.arange(its), np.transpose(np.array(trajectories)), color="lightgrey")
    if true is not None:
        plt.plot(np.arange(its), np.transpose(true), color = "black", label='Ensemble mean', linewidth=0.8)
    plt.legend(loc = "upper right")
    ax.set_xlabel('Time (generations)', size=14)
    ax.set_ylabel('Population size', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig('plots/trajectories.png')

def plot_forecast(observations, historic_mean, ricker, its, test_index, pars,  phi = None, var=None):

    fig = plt.figure()
    ax = fig.add_subplot()
    obs = np.transpose(np.array(observations))
    plt.plot(np.arange(obs.shape[0]), obs, color="grey", label ='observations')
    plt.plot(np.arange(test_index,its), np.transpose(np.array(historic_mean)), color="red", label='historic mean')
    if not var is None:
        ax.fill_between(np.arange(test_index,its), np.transpose(np.array(historic_mean+var)), np.transpose(np.array(historic_mean-var)),
                        color="red", alpha=0.3)
    plt.plot(np.arange(its), np.transpose(np.array(ricker)), color="blue", alpha=0.5)
    plt.axvline(x=test_index, color="black", linestyle="--")
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (generations)', size=14)
    ax.set_ylabel('Population size', size=14)
    ax.set_title(phi)
    fig.show()
    fig.savefig(f'plots/forecast_{pars}.png')

def FP_rmse(mat, fpt, pars, mat2 = None, phi= None):

    fig = plt.figure()
    ax = fig.add_subplot()
    fc = np.array(mat)
    fc_mean = np.mean(fc, axis=0)
    fc_std = np.std(fc)
    plt.plot(np.arange(fc.shape[1]), fc_mean, label="Estimated", color='green')
    ax.fill_between(np.arange(fc.shape[1]), fc_mean+fc_std, fc_mean-fc_std, color="green", alpha=0.2)
    if not mat2 is None:
        fc2 = np.array(mat2)
        fc_mean2 = np.mean(fc2, axis=0)
        fc_std2 = np.std(fc2)
        plt.plot(np.arange(fc2.shape[1]), fc_mean2, label="Perfect", color='blue')
        ax.fill_between(np.arange(fc2.shape[1]), fc_mean2 + fc_std2, fc_mean2 - fc_std2, color="blue", alpha=0.15)
    plt.axhline(y=fpt, color="black", linestyle="--", label="Historic mean")
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Root mean square error')
    #ax.set_ylim([0, 20])
    ax.legend(loc='lower right')
    fig.show()
    fig.savefig(f'plots/forecast_error_distributions_{pars}.png')

def FP_correlation(mat, fpt, pars, mat2 = None, phi = None):

    fig = plt.figure()
    ax = fig.add_subplot()
    fc = np.round(np.array(mat), 4)
    fc_mean = np.mean(fc, axis=0)
    fc_std = np.std(fc, axis=0)
    plt.plot(np.arange(fc.shape[1]), fc_mean, label="Estimated", color='green')
    ax.fill_between(np.arange(fc.shape[1]), fc_mean+fc_std, fc_mean-fc_std, color="green", alpha=0.2)

    if not mat2 is None:
        fc2 = np.round(np.array(mat2), 4)
        fc_mean2 = np.mean(fc2, axis=0)
        fc_std2 = np.std(fc2, axis=0)
        plt.plot(np.arange(fc2.shape[1]), fc_mean2, label="Perfect Model", color='blue')
        ax.fill_between(np.arange(fc2.shape[1]), fc_mean2 + fc_std2, fc_mean2 - fc_std2, color="blue", alpha=0.15)

    plt.axhline(y=fpt, color="black", linestyle="--", label="Forecast Proficiency threshold")
    ax.set_xlabel('Time (generations)', size=14)
    ax.set_ylabel('Pearsons r', size=14)
    #ax.set_ylim([0, 20])
    ax.legend(loc='lower left', prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(phi)
    fig.show()
    fig.savefig(f'plots/forecast_corr_distributions_{pars}.png')

def FP_absdifferences(absolute_differences, absolute_differences_mean, its, log=False,
                      dir = 'plots/baseline/proficiencies'):

    fig = plt.figure()
    ax = fig.add_subplot()
    if log:
        plt.plot(np.arange(its), np.transpose(np.log(absolute_differences)), color="lightgrey")
        plt.plot(np.arange(its), np.transpose(np.log(absolute_differences_mean)), color="black", label="Ensemble mean")
        ax.set_ylabel("Log of absolute difference to truth", size=14)
    else:
        plt.plot(np.arange(its), np.transpose(absolute_differences), color="lightgrey")
        plt.plot(np.arange(its), np.transpose(absolute_differences_mean), color="black", label="Ensemble mean")
        ax.set_ylabel("Absolute difference to truth", size=14)
    ax.set_xlabel("Time step", size=14)
    ax.legend(loc="upper left", prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig(f'{dir}/FP_absdifferences.png')

def FP_correlation(corrs, dir = 'plots/baseline/proficiencies'):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(np.transpose(corrs), color="lightgrey")
    plt.plot(np.mean(np.transpose(corrs), axis=1), color="black", label="Ensemble mean")
    ax.set_ylabel("Pearson's r", size=14)
    ax.set_xlabel("Time step", size=14)
    ax.legend(loc="lower left", prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig(f'{dir}/FP_correlations.png')

def plot_mean_efh_varying_thresholds(measure, efhs, threshold_seq, title, legend_position=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(threshold_seq, efhs, color="lightblue")
    plt.plot(threshold_seq, np.mean(efhs, axis=1), color="darkblue", label = "Optimal FH")
    ax.set_xlabel(f'{title} threshold for acceptance', size=14)
    ax.set_ylabel('Predicted forecast horizon', size=14)
    if not legend_position is None:
        ax.legend(loc=legend_position, prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig(f'plots/baseline/efh_mean/{measure}_threshold_m.png')

def plot_quantile_efh(measure, efh_corrs_ps, threshold_seq, title, label = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(threshold_seq, efh_corrs_ps, color="darkblue")
    plt.plot(threshold_seq, np.mean(efh_corrs_ps, axis=1), color="yellow", label=label)
    ax.set_xlabel(f'{title} threshold for acceptance', size=14)
    ax.set_ylabel('Predicted forecast horizon', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if not label is None:
        ax.legend(loc="lower right", prop={"size": 14})
    fig.show()
    fig.savefig(f'plots/baseline/efh_quantile/{measure}_threshold_qu.png')

def plot_efh_varying_thresholds_HP(metric, efhs_mcorrs, ehfs_mcorrs_m, threshold_seq, ensemble_size):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(ensemble_size):
        plt.plot(threshold_seq, efhs_mcorrs[:,:,i], color="lightblue")
    plt.plot(threshold_seq, ehfs_mcorrs_m, color="darkblue")
    plt.plot(threshold_seq, np.mean(ehfs_mcorrs_m, axis=1), color="yellow", label = "Mean potential FH")
    plt.plot(threshold_seq, np.max(ehfs_mcorrs_m, axis=1), color="orange", label = "Max potential FH")
    plt.axvline(x=0.5, linestyle="dashed", color="grey", linewidth=1.5)
    ax.set_xlabel('Correlation threshold for acceptance', size=14)
    ax.set_ylabel('Predicted forecast horizon', size=14)
    ax.legend(loc="lower left", prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig(f'plots/baseline/efh_mean/corr_threshold_window.png')

def plot_lyapunov_exponents(r_values, true_lyapunovs, predicted_lyapunovs):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(r_values, predicted_lyapunovs, color="red")
    plt.plot(r_values, true_lyapunovs, label = "true", color="blue")
    ax.set_ylabel("Lyapunov exponent", size=14)
    ax.set_xlabel("r value", size=14)
    ax.legend(loc="lower right")
    fig.show()

def plot_ln_deltaRatio(delta_range, Delta_range, l):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(l):
        plt.plot(delta_range, np.log(Delta_range[i]/delta_range), color = "lightgrey")
    ax.set_ylabel('ln($\Delta/\delta_0$)', size=14)
    ax.set_xlabel('$\delta_0$-range', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/ln_deltaRatio.png')
    fig.show()

def delta_U(Delta_range, predicted_efh, predicted_efh_m, ensemble_size):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(ensemble_size):
        plt.plot(Delta_range, predicted_efh[:,:,i], color="lightblue")
    plt.plot(Delta_range, predicted_efh_m, color="darkblue")
    plt.plot(Delta_range, np.mean(predicted_efh_m, axis=1), color="yellow", label = "Mean potential FH")
    plt.plot(Delta_range, np.max(predicted_efh_m, axis=1), color="orange", label = "Max potential FH")
    ax.set_xlabel('Forecast proficiency threshold ($\Delta$)', size=14)
    ax.set_ylabel('Predicted forecast horizon', size=14)
    ax.legend(loc="lower right", prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/delta_U.png')
    fig.show()

def delta_L(delta_range, predicted_efh, predicted_efh_m, ensemble_size):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(ensemble_size):
        plt.plot(delta_range, predicted_efh[:,:,i], color="lightblue")
    plt.plot(delta_range, predicted_efh_m, color="darkblue")
    plt.plot(delta_range, np.mean(predicted_efh_m, axis=1), color="yellow", label = "Mean potential FH")
    plt.plot(delta_range, np.max(predicted_efh_m, axis=1), color="orange", label= "Max potential FH")
    ax.set_xlabel('Initial uncertainty ($\delta_0$)', size=14)
    ax.set_ylabel('Predicted forecast horizon', size=14)
    ax.legend(loc="upper right", prop = {"size":14})
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/delta_L.png')
    fig.show()

def potential_fh_roc(potential_fh_roc, delta_range):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(delta_range[1:], np.transpose(potential_fh_roc))
    ax.set_xlabel('Initial uncertainty ($\delta_0$)', size=14)
    ax.set_ylabel('Rate of change: Max potential FH', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/potential_fh_roc.png')
    fig.show()


def fixed_Tp_delta(delta_range,Delta_range, tp_Delta, tp_delta):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(delta_range, np.transpose(tp_Delta), color="lightgrey")
    plt.plot(delta_range, np.mean(np.transpose(tp_Delta), axis=1), color="black", label="Ensemble mean")
    ax.set_xlabel('Initial uncertainty ($\delta_0$)', size=14)
    ax.set_ylabel('Acceptable $\Delta$ for FH of 20', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/fixed_tp_Delta.png')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(Delta_range, np.transpose(tp_delta), color="lightgrey")
    plt.plot(Delta_range, np.mean(np.transpose(tp_delta), axis=1), color="black", label="Ensemble mean")
    ax.set_xlabel('Forecast proficiency ($\Delta$)', size=14)
    ax.set_ylabel('Acceptable $\delta$ for FH of 20', size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(r'plots/baseline/efh_lyapunov/fixed_tp_delta.png')
    fig.show()

def lyapunov_time(predicted_efhs_ms, r_values):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(r_values, predicted_efhs_ms, color="lightgrey")
    plt.plot(r_values, np.mean(predicted_efhs_ms, axis=1), color="blue", label="Ensemble mean")
    ax.set_xlabel('Growth rate r', size=14)
    ax.set_ylabel('Predicted Lyapunov time', size=14)
    ax.axhline(y=0, linestyle="dashed", color="grey")
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc="upper left", prop = {"size":14})
    fig.show()
    fig.savefig(r'plots/baseline/efh_lyapunov/lyapunov_time.png')

def lyapunov_time_modifier_effect(r_values, predicted_efhs_fix1_ms, predicted_efhs_fix2_ms):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(r_values, predicted_efhs_fix1_ms, color="lightgreen", label = "$ln(\dfrac{\Delta}{\delta_0})$ < 1")
    plt.plot(r_values, predicted_efhs_fix2_ms, color="lightblue", label = "$ln(\dfrac{\Delta}{\delta_0})$ > 1")
    ax.set_xlabel('Growth rate r', size=14)
    ax.set_ylabel('Predicted Lyapunov time', size=14)
    ax.axhline(y=0, linestyle="dashed", color="grey")
    ax.tick_params(axis='both', which='major', labelsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    ax.legend(newHandles, newLabels, loc="lower left", prop = {"size":14})
    fig.show()
    fig.savefig(r'plots/baseline/efh_lyapunov/lyapunov_time_modifier_effect.png')

def lyapunovs_along_r(r_values, lyapunovs):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(r_values, lyapunovs, color="lightgrey")
    plt.plot(r_values, np.mean(lyapunovs, axis=1), color="blue")
    ax.set_xlabel('Growth rate r', size=14)
    ax.set_ylabel('Lyapunov exponent', size=14)
    ax.axhline(y=0, linestyle="dashed", color="grey")
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.show()
    fig.savefig(r'plots/baseline/efh_lyapunov/lyapunovs_along_r.png')