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
    plt.plot(np.arange(31,50), np.transpose(np.array(historic_mean)), color="red", label='historic mean')
    if not var is None:
        ax.fill_between(np.arange(31,50), np.transpose(np.array(historic_mean+var)), np.transpose(np.array(historic_mean-var)),
                        color="red", alpha=0.5)
    plt.plot(np.arange(its), np.transpose(np.array(ricker)), color="blue", label='ricker', alpha=0.5)
    plt.axvline(x=31, color="black", linestyle="--")
    plt.legend(loc='upper left')
    ax.set_xlabel('Time (generations)')
    ax.set_ylabel('Population size')
    ax.set_title(phi)
    fig.show()
    fig.savefig('plots/forecast.png')