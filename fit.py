import numpy as np
import scipy.optimize as optim
import models
from vizualisations import baseplot
import utils

# function to minimize: Residuals
def lsq_fit(mod, x_train, bounds = (0, [4.])):
    """
    Fits the ricker model to data in x_train.
    Currently just estimating r. Enable for sigma!
    :param x_train: 1d array, timeseries of true dynamics.
    :param bounds: boundaries for paramter values to sample within
    :return: least squares object
    """

    def fun(pars, x, y):
        mod.theta['lambda_a'] = pars[0]
        mod.theta['lambda_b'] = pars[1]
        mod.theta['beta'] = pars[2]
        mod.theta['delta'] = pars[3]
        res = mod.model(x, fit=True) - y
        return res

    # Initialize parameters randomly
    lambda_init = np.random.normal(mod.theta['lambda_a'], 0.01, 1)[0]
    alpha_init = np.random.normal(mod.theta['lambda_b'], 0.01, 1)[0]
    beta_init = np.random.normal(mod.theta['beta'], 0.01, 1)[0]
    delta_init = np.random.normal(mod.theta['delta'], 0.01, 1)[0]
    p0 = [lambda_init, alpha_init, beta_init, delta_init]

    # Data
    x = x_train[:-1]
    y = x_train[1:]

    lsq_solution = optim.least_squares(fun, p0, bounds = (-0.1, 0.1), loss = 'soft_l1', args=(x, y))
    lam = lsq_solution.x[0]
    alpha = lsq_solution.x[1]
    beta = lsq_solution.x[2]
    delta = lsq_solution.x[3]

    theta_hat = {'lambda_a':lam, 'lambda_b':alpha, 'beta':beta, 'delta':delta}
    print(theta_hat)

    return theta_hat, [lam, alpha, beta, delta]


uq = {"parameters":False, "initial":True,"process":True}
hp = {"iterations": 2 * 52, "initial_size": 950,
               "initial_uncertainty": 1e-3, "ensemble_size": 10}
theta = {'lambda': 0.005, 'alpha': 1 / 1000, 'sigma': 0.1}

ricker = models.Ricker_Single(uncertainties=uq)
ricker.set_parameters(theta)
simu = ricker.simulate(hp)
x = simu["ts"]

baseplot(x, transpose=True)

x_train = x[1]

ricker2 = models.Ricker_Multi(uncertainties=uq)
theta2 = {'lambda_a': 0.05, 'alpha':1/1000, 'beta':1/1950,
          'lambda_b': 0.05, 'gamma': 1/1000, 'delta':1/1955,
          'sigma':0.1}
ricker2.set_parameters(theta2)

fit, fit_ls = lsq_fit(ricker2, x_train)

theta_hat = {'lambda_a': fit_ls[0], 'alpha':1/1000, 'beta':fit_ls[2],
          'lambda_b': fit_ls[1], 'gamma': 1/1000, 'delta':fit_ls[3],
          'sigma':0.1}
hp = {"iterations": 5 * 52, "initial_size": (950, 950),
               "initial_uncertainty": 1e-3, "ensemble_size": 10}

ricker2.set_parameters(theta_hat)
simu = ricker2.simulate(hp, derive=False)
x = simu['ts']

baseplot(x[:,:,0], transpose=True)


## the same with exogeneous
# use fitted params from above.



def lsq_fit(mod, x_train, T, bounds = (0, [4.])):
    """
    Fits the ricker model to data in x_train.
    Currently just estimating r. Enable for sigma!
    :param x_train: 1d array, timeseries of true dynamics.
    :param bounds: boundaries for paramter values to sample within
    :return: least squares object
    """

    def fun(pars, x, y, T):
        mod.theta_upper['ax'] = pars[0]
        mod.theta_upper['ay'] = pars[1]
        mod.theta_upper['bx'] = pars[2]
        mod.theta_upper['cx'] = pars[3]
        mod.theta_upper['by'] = pars[4]
        mod.theta_upper['cy'] = pars[5]
        res = mod.model(x, T, fit=True) - y
        return res

    # Initialize parameters randomly
    ax_init = np.random.normal(mod.theta_upper['ax'], 0.01, 1)[0]
    ay_init = np.random.normal(mod.theta_upper['ay'], 0.01, 1)[0]
    lambda_init = np.random.normal(mod.theta_upper['bx'], 0.01, 1)[0]
    alpha_init = np.random.normal(mod.theta_upper['cx'], 0.01, 1)[0]
    beta_init = np.random.normal(mod.theta_upper['by'], 0.01, 1)[0]
    delta_init = np.random.normal(mod.theta_upper['cy'], 0.01, 1)[0]
    p0 = [ax_init, ay_init, lambda_init, alpha_init, beta_init, delta_init]

    # Data
    x = x_train[:-1]
    y = x_train[1:]

    lsq_solution = optim.least_squares(fun, p0, bounds = (0.00, 0.5), loss = 'soft_l1', args=(x, y, T))
    ax = lsq_solution.x[0]
    ay = lsq_solution.x[1]
    lam = lsq_solution.x[2]
    alpha = lsq_solution.x[3]
    beta = lsq_solution.x[4]
    delta = lsq_solution.x[5]

    theta_hat = {'ax':ax, 'ay':ay, 'bx':lam, 'cx':alpha, 'by':beta, 'cy':delta}
    print(theta_hat)

    return theta_hat, [ax, ay, lam, alpha, beta, delta]

uq = {"parameters":False, "initial":True,"process":True}
hp = {"iterations": 2 * 52, "initial_size": 950,
               "initial_uncertainty": 1e-3, "ensemble_size": 10}
theta = {'alpha': 1 / 1000, 'sigma': 0.1}
theta_upper = {'ax': 0.005, 'bx': .05, 'cx': .03}

T = utils.simulate_T(hp['iterations'], add_trend=False, add_noise=True, show=True)
ricker = models.Ricker_Single_T(uncertainties=uq)
ricker.set_parameters(theta, theta_upper)
simu = ricker.simulate(hp, derive =False, ex=T)
x = simu["ts"]
baseplot(x, transpose=True)

x_train = x[1]

ricker2 = models.Ricker_Multi_T(uncertainties=uq)
theta = {'alpha':1/1000, 'beta':fit_ls[2],
           'gamma': 1/1000, 'delta':fit_ls[3],
          'sigma':0.1}
theta_upper = {'ax': fit_ls[0], 'bx': 0.08, 'cx': 0.05,
                'ay': fit_ls[1], 'by': 0.08, 'cy':0.05}

ricker2.set_parameters(theta, theta_upper)

fit2, fit_ls2 = lsq_fit(ricker2, x_train, T[:-1])

theta_upper_hat = {'ax': fit_ls2[0], 'bx': fit_ls2[0], 'cx': fit_ls2[1],
                'ay': fit_ls2[1], 'by': fit_ls2[2], 'cy': fit_ls2[3]}
hp = {"iterations": 4 * 52, "initial_size": (950,950),
               "initial_uncertainty": 1e-3, "ensemble_size": 10}
T = utils.simulate_T(hp['iterations'], add_trend=False, add_noise=True, show=True)

ricker2.set_parameters(theta, theta_upper_hat)
simu = ricker2.simulate(hp, derive =False, ex=T)
x1 = simu["ts"][:,:,0]
x2 = simu["ts"][:,:,1]
baseplot(x, transpose=True)