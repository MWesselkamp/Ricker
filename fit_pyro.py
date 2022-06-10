import matplotlib
matplotlib.use('module://backend_interagg')
import model
from vizualisations import plot_posterior
from vizualisations import plot_trajectories
import pickle
import numpy as np

if __name__ == '__main__':

    r_real = 1.5
    sd_real = 0.001
    N0_mean = 0.8
    N0_sd = 0
    mu = 0
    sigma = 0.005

    params = {'N0_mean': N0_mean, 'N0_sd': N0_sd, 'r_mean': r_real, 'r_sd': sd_real, 'k': 1}

    x, trues = model.ricker_simulate(1, 50, params, stoch = False)
    np.savetxt("data/realdynamics.csv", x, delimiter=',')

    x_train = x[:, :30]
    x_test = x[:, 31:]

    mod = model.Ricker()  # Create class instance
    mcmc, post_samples = mod.fit_model_pyro(x_train)
    fit = open('results/fit.pkl', 'wb')
    pickle.dump(post_samples, fit)
    fit.close()

    post_samples = open("results/fit.pkl", "rb")
    post_samples = pickle.load(post_samples)

    r_posterior = post_samples['r'].numpy()
    sigma_posterior = post_samples['sigma'].numpy()

    plot_posterior(post_samples['r'])
    plot_posterior(post_samples['sigma'])

    trajectories = mod.model_iterate(N0_mean, r_posterior, iterations=50, sigma = np.mean(sigma_posterior))
    plot_trajectories(np.transpose(trajectories), x)
