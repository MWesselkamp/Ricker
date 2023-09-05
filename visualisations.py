import matplotlib.pyplot as plt
import numpy as np
from CRPS import CRPS
from metrics import mse
import torch
import os

def plot_fit(ypreds, y, scenario, process, clim = None, save=True):

    fh_metric1 = {'MSE': mse(y.detach().numpy()[np.newaxis, :], ypreds),
                     'MSE_clim': mse(y.detach().numpy()[np.newaxis, :], clim.detach().numpy())}
    fh_metric2 = {
        'CRPS': [CRPS(ypreds[:, i], y.detach().numpy()[i]).compute()[0] for i in range(ypreds.shape[1])],
        'CRPS_clim': [CRPS(clim.detach().numpy()[:, i], y.detach().numpy()[i]).compute()[0] for i in
                      range(ypreds.shape[1])]}

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3,0.8,0.8]})
    if not clim is None:
        clim = ax1.plot(np.transpose(clim.detach().numpy()), color='gray', label = 'Long-term\n Mean', zorder=0)
    true = ax1.plot(np.transpose(y.detach().numpy()), color='red', label='Observed', zorder=1)
    #fit = ax1.plot(np.transpose(ypreds), color='blue', label='Fitted', alpha=0.5)
    fit = ax1.fill_between(np.arange(ypreds.shape[1]), ypreds.transpose().min(axis=1),
                          ypreds.transpose().max(axis=1), color='b', alpha=0.4)
    fit_mean = ax1.plot(ypreds.transpose().mean(axis=1), color='b', alpha=0.5, label='Ricker')
    if not clim is None:
        plt.setp(clim[1:], label="_")
    #plt.setp(fit_mean[1:], label="_")
    plt.setp(true[1:], label="_")
    ax1.legend()
    ax1.set_ylabel('Relative size')
    if not fh_metric1 is None:
        ax2.plot(list(fh_metric1.values())[1], color='gray', linewidth=0.8)
        ax2.plot(list(fh_metric1.values())[0], color='blue', linewidth=0.8)
        ax2.set_ylabel(list(fh_metric1.keys())[0])
    else:
        ax2.plot(np.transpose(ypreds) - np.transpose(y.detach().numpy()[np.newaxis, :]), color='gray', linewidth=0.8)
        ax2.set_ylabel('Absolute error')
    ax2.axhline(y=0, color = 'black', linestyle='--', linewidth = 0.8)
    #ax2.set_xlabel('Timestep [Days]')
    if not fh_metric2 is None:
        ax3.plot(list(fh_metric2.values())[1], color='gray', linewidth=0.8)
        ax3.plot(list(fh_metric2.values())[0], color='blue', linewidth=0.8)
        ax3.set_ylabel(list(fh_metric2.keys())[0])
    else:
        ax3.plot(np.transpose(ypreds) - np.transpose(y.detach().numpy()[np.newaxis, :]), color='gray', linewidth=0.8)
        ax3.set_ylabel('Absolute error')
    ax3.axhline(y=0, color = 'black', linestyle='--', linewidth = 0.8)
    ax3.set_xlabel('Generation')
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{scenario}_{process}/verification_setting.pdf')

def plot_fit2(ypreds, y, scenario, process, clim = None, fhs = None, save=True):


    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3,1]})
    if not clim is None:
        clim = ax1.plot(np.transpose(clim.detach().numpy()), color='gray', label = 'Long-term\n Mean', zorder=0)
    true = ax1.plot(np.transpose(y.detach().numpy()), color='red', label='Observed', zorder=1)
    #fit = ax1.plot(np.transpose(ypreds), color='blue', label='Fitted', alpha=0.5)
    fit = ax1.fill_between(np.arange(ypreds.shape[1]), ypreds.transpose().min(axis=1),
                          ypreds.transpose().max(axis=1), color='b', alpha=0.4)
    fit_mean = ax1.plot(ypreds.transpose().mean(axis=1), color='b', alpha=0.5, label='Ricker')
    if not clim is None:
        plt.setp(clim[1:], label="_")

    #plt.setp(fit_mean[1:], label="_")
    plt.setp(true[1:], label="_")
    ax1.legend()
    ax1.set_ylabel('Relative size')
    ax2.plot(np.transpose(ypreds) - np.transpose(y.detach().numpy()[np.newaxis, :]), color='gray', linewidth=0.8)
    if not fhs is None:
        fh_ls = ['--', '-']
        i = 0
        for fh in fhs:
            ax2.vlines(fh, ymin=-1, ymax=1, linestyles=fh_ls[i], color = 'black')
            i += 1
    ax2.set_ylabel('Residuals')
    ax2.axhline(y=0, color = 'black', linestyle='--', linewidth = 0.8)
    ax2.set_xlabel('Generation')
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{scenario}_{process}/verification_setting_anomaly.pdf')


def plot_all_dynamics(obs, preds, ref, save):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharey=True, sharex=True)

    ax[0, 0].plot(ref[1].transpose(), color='gray', alpha=0.9, zorder=0)
    ax[0, 0].plot(obs[1].transpose(), color='red', alpha=0.8, zorder=1)
    ax[0, 0].fill_between(np.arange(preds[1].shape[1]), preds[1].transpose().min(axis=1),
                          preds[1].transpose().max(axis=1), color='b', alpha=0.4)
    ax[0, 0].plot(preds[1].transpose().mean(axis=1), color='b', alpha=0.5)
    ax[0, 0].set_ylabel('Relative size')

    l1 = ax[0, 1].plot(ref[0].transpose(), color='gray', label='Climatology', alpha=0.9, zorder=0)
    l2 = ax[0, 1].plot(obs[0].transpose(), color='red', label='Observed', alpha=0.8, zorder=1)
    l3 = ax[0, 1].fill_between(np.arange(preds[0].shape[1]), preds[0].transpose().min(axis=1),
                               preds[0].transpose().max(axis=1), color='b', alpha=0.4)
    l4 = ax[0, 1].plot(preds[0].transpose().mean(axis=1), color='b', alpha=0.5, label='Ricker')

    ax[1, 0].plot(ref[3].transpose(), color='gray', zorder=0)
    ax[1, 0].plot(obs[3].transpose(), color='red', alpha=0.6, zorder=1)
    ax[1, 0].fill_between(np.arange(preds[3].shape[1]), preds[3].transpose().min(axis=1),
                          preds[3].transpose().max(axis=1), color='b', alpha=0.4)
    ax[1, 0].plot(preds[3].transpose().mean(axis=1), color='b', alpha=0.5)
    ax[1, 0].set_ylabel('Relative size')
    ax[1, 0].set_xlabel('Time index')

    ax[1, 1].plot(ref[2].transpose(), color='gray', zorder=0)
    ax[1, 1].plot(obs[2].transpose(), color='red', alpha=0.6, zorder=1)
    ax[1, 1].fill_between(np.arange(preds[2].shape[1]), preds[2].transpose().min(axis=1),
                          preds[2].transpose().max(axis=1), color='b', alpha=0.4)
    ax[1, 1].plot(preds[2].transpose().mean(axis=1), color='b', alpha=0.5)
    ax[1, 1].set_xlabel('Time index')

    plt.setp(l1[1:], label="_")
    plt.setp(l2[1:], label="_")
    plt.setp(l4[1:], label="_")
    ax[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.show()

    if save:
        plt.savefig(f'results/dynamics_all.pdf')

def plot_horizons(fha_ricker,fha_reference, fhp_ricker, fhp_reference, fsh, metrics_fh, scenario, process, save, show_upper = False):
    plt.figure(figsize=(9,6))
    x_positions = np.arange(len(metrics_fh))
    x_labels = ['Corr', 'MAE', 'F-Stats', 'CRPS(S)'] # ['Corr', 'MSE', 'MAE', 'CRPS']
    x_colors = ['gray', 'gray', 'silver', 'lightgray']
    for i in range(len(x_positions)):
        plt.axvspan(x_positions[i]-0.5, x_positions[i]+1-0.5, facecolor=x_colors[i], alpha=0.5)
    plt.hlines(xmin=min(x_positions)-0.5, xmax=max(x_positions)+0.5, y = 0, linestyles='--', colors='black', linewidth = 0.5)
    plt.scatter(x_positions-0.2, fha_ricker, marker='D',s=120, color='blue', label='$h_{Ricker}$')
    plt.scatter(x_positions, fha_reference, marker='D',s=120, color='green', label='$h_{Climatology}$ ')
    plt.scatter(x_positions-0.2, fhp_ricker, marker='D',s=120, facecolors='none', edgecolors='blue', label='$\hat{h}_{Ricker}$')
    plt.scatter(x_positions, fhp_reference, marker='D',s=120, facecolors='none', edgecolors='green', label='$\hat{h}_{Climatology}$')
    plt.scatter(x_positions+0.2, fsh, marker='D',s=120, color='red', label='$h_{skill}$')
    plt.ylabel('Forecast horizon', fontweight='bold')
    plt.xlabel('Proficiency', fontweight='bold')
    plt.xticks(x_positions, x_labels)
    if show_upper:
        plt.ylim((-10,380))
    else:
        plt.ylim((-3,110))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save:
        plt.savefig(f'results/{scenario}_{process}/horizons.pdf')

def plot_horizon_maps(mat_ricker, mat_climatology, mat_ricker_perfect, scenario, process, save=True):
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), sharey=False, sharex=False)
    custom_min = 0
    custom_max = 0.6

    heatmap0 = ax[0, 0].imshow(mat_ricker.transpose()[0:, :], vmin=custom_min, vmax=custom_max, cmap='plasma_r')
    cb = plt.colorbar(heatmap0, ax=ax[0, 0])
    cb.set_label('CRPS')
    ax[0, 0].set_ylabel('Day of forecast initialization')
    ax[0, 0].set_xlabel('Forecast time')
    ax[0, 0].autoscale(False)

    ax[0, 1].imshow((mat_ricker.transpose()[0:, :] > 0.05), cmap='plasma_r')
    # plt.colorbar()
    ax[0, 1].set_ylabel('Day of forecast initialization')
    ax[0, 1].set_xlabel('Forecast time')

    ax[0, 2].imshow((mat_climatology.transpose()[0:, :] - mat_ricker.transpose()[0:, :]) < -0.05, cmap='plasma_r')
    # ax[0,2].colorbar()
    ax[0, 2].set_ylabel('Day of forecast initialization')
    ax[0, 2].set_xlabel('Forecast time')

    heatmap1 = ax[1, 0].imshow(mat_ricker_perfect.transpose()[0:, :], vmin=custom_min, vmax=custom_max, cmap='plasma_r')
    cb1 = plt.colorbar(heatmap1, ax=ax[1, 0])
    cb1.set_label('CRPS')
    ax[1, 0].set_ylabel('Day of forecast initialization')
    ax[1, 0].set_xlabel('Forecast time')
    ax[1, 0].autoscale(False)

    ax[1, 1].imshow((mat_ricker_perfect.transpose()[0:, :] > 0.05), cmap='plasma_r')
    # ax[0,0].colorbar()
    ax[1, 1].set_ylabel('Day of forecast initialization')
    ax[1, 1].set_xlabel('Forecast time')

    fig.delaxes(ax[1, 2])
    plt.tight_layout()

    if save:
        plt.savefig(f'results/{scenario}_{process}/horizon_maps.pdf')

def plot_losses(losses, loss_fun, log=True, saveto=''):
    if log:
        ll = np.log(torch.stack(losses).detach().numpy())
    else:
        ll = torch.stack(losses).detach().numpy()
    plt.plot(ll)
    plt.ylabel(f'{loss_fun} loss')
    plt.xlabel(f'Epoch')
    plt.savefig(os.path.join(saveto, 'losses.pdf'))
    plt.close()

def plot_posterior(df, saveto=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes[0, 0].hist(df["alpha"], bins=10, edgecolor='black')
    axes[0, 0].set_title("Histogram of alpha")
    axes[0, 1].hist(df["beta"], bins=10, edgecolor='black')
    axes[0, 1].set_title("Histogram of beta")
    axes[1, 0].hist(df["bx"], bins=10, edgecolor='black')
    axes[1, 0].set_title("Histogram of bx")
    axes[1, 1].hist(df["cx"], bins=10, edgecolor='black')
    axes[1, 1].set_title("Histogram of cx")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(saveto, 'posterior.pdf'))
    plt.close()


def baseplot(x1, x2=None, x3 = None, transpose=False, xlab=None, ylab=None):
    if transpose:
        x1 = np.transpose(x1)
        if not x2 is None:
            x2 = np.transpose(x2)
        if not x3 is None:
            x3 = np.transpose(x3)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x1, alpha=0.5, color="blue")
    if not x2 is None:
        ax.plot(x2, alpha=0.5, color="red")
    if not x3 is None:
        ax.plot(x3, alpha=0.5, color="green")
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