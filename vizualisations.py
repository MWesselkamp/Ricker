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

def plot_forecast(observations, historic_mean, ricker, its, phi = None, var=None):

    fig = plt.figure()
    ax = fig.add_subplot()
    obs = np.transpose(np.array(observations))
    plt.plot(np.arange(obs.shape[0]), obs, color="grey", label ='observations')
    plt.plot(np.arange(31,its), np.transpose(np.array(historic_mean)), color="red", label='historic mean')
    if not var is None:
        ax.fill_between(np.arange(31,its), np.transpose(np.array(historic_mean+var)), np.transpose(np.array(historic_mean-var)),
                        color="red", alpha=0.3)
    plt.plot(np.arange(its), np.transpose(np.array(ricker)), color="blue", alpha=0.5)
    plt.axvline(x=31, color="black", linestyle="--")
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
    ax.set_title(phi)
    fig.show()
    fig.savefig('plots/forecast.png')

def forecast_error_distributions(mat, fpt):

    fig = plt.figure()
    ax = fig.add_subplot()
    fc = np.array(mat)
    fc_mean = np.mean(fc, axis=1)
    fc_std = np.std(fc)
    plt.plot(np.arange(fc.shape[0]), fc_mean, label="Ricker", color='blue')
    ax.fill_between(np.arange(fc.shape[0]), fc_mean+fc_std, fc_mean-fc_std, color="blue", alpha=0.3)
    plt.axhline(y=fpt, color="black", linestyle="--", label="Historic mean")
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Root mean square error')
    #ax.set_ylim([0, 20])
    ax.legend(loc='lower right')
    fig.show()
    fig.savefig('plots/forecast_error_distributions.png')

def forecast_corr_distributions(mat):

    fig = plt.figure()
    ax = fig.add_subplot()
    fc = np.round(np.array(mat), 4)
    fc_mean = np.mean(fc, axis=0)
    fc_std = np.std(fc, axis=0)
    print(fc_mean)
    print(np.log(fc_mean))
    plt.plot(np.arange(fc.shape[1]), fc_mean, label="Ricker", color='blue')
    #ax.fill_between(np.arange(fc.shape[1]), fc_mean+fc_std, fc_mean-fc_std, color="blue", alpha=0.3)
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Pearsons r')
    #ax.set_ylim([0, 20])
    ax.legend(loc='lower right')
    fig.show()
    fig.savefig('plots/forecast_corr_distributions.png')