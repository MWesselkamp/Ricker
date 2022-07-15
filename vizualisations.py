import matplotlib.pyplot as plt
import numpy as np

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
    plt.plot(np.arange(its), np.transpose(np.array(trajectories)), color="grey")
    if true is not None:
        plt.plot(np.arange(its), np.transpose(true), color = "black", label='observations')
    plt.legend()
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
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
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
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
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Pearsons r')
    #ax.set_ylim([0, 20])
    ax.legend(loc='lower left')
    ax.set_title(phi)
    fig.show()
    fig.savefig(f'plots/forecast_corr_distributions_{pars}.png')

def FP_absdifferences(absolute_differences, absolute_differences_mean, its, log=False,
                      dir = 'plots/baseline/proficiencies'):

    fig = plt.figure()
    ax = fig.add_subplot()
    if log:
        plt.plot(np.arange(its), np.transpose(np.log(absolute_differences)), color="grey")
        plt.plot(np.arange(its), np.transpose(np.log(absolute_differences_mean)), color="black", label="mean")
        ax.set_ylabel("Log of absolute difference to truth")
    else:
        plt.plot(np.arange(its), np.transpose(absolute_differences), color="grey")
        plt.plot(np.arange(its), np.transpose(absolute_differences_mean), color="black", label="mean")
        ax.set_ylabel("Absolute difference to truth")
    ax.set_xlabel("Time step")
    fig.show()
    fig.savefig(f'{dir}/FP_absdifferences.png')


def plot_lyapunov_exponents(r_values, true_lyapunovs, predicted_lyapunovs):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(r_values, predicted_lyapunovs, color="red")
    plt.plot(r_values, true_lyapunovs, label = "true", color="blue")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_xlabel("r value")
    ax.legend(loc="lower right")
    fig.show()

def plot_lyapunov_efhs(r_values, predicted_efhs, log=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    if log:
        plt.plot(r_values, predicted_efhs, color="blue")
        ax.set_ylabel("Forecast horizon")
    else:
        plt.plot(r_values, np.log10(predicted_efhs), color="blue")
        ax.set_ylabel("Log10(Forecast horizon)")
    ax.set_xlabel("r value")
    fig.show()

def plot_LE_efh_along_Delta(Delta_range, predicted_efh):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(Delta_range, predicted_efh, color="gray")
    plt.plot(Delta_range, np.mean(predicted_efh, axis=1), color="black")
    ax.set_xlabel('Forecast proficiency threshold ($\Delta$)')
    ax.set_ylabel('Predicted forecast horizons')
    fig.show()