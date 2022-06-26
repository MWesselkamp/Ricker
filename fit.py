import numpy as np
import scipy.optimize as optim

import model

# function to minimize: Residuals
def lsq_fit(x_train, theta_true):
    """
    Fits the ricker model to data in x_train.
    At the moment only optimizes for phi and log_r but not for sigma.
    :param x_train:
    :param theta_true:
    :return:
    """

    def fun(pars, x, y):
        res = model.ricker(x, pars[0]) - y
        return res

    # Initialize parameters randomly
    r_init = np.random.normal(theta_true['r'], 0.5, 1)[0]
    p0 = [r_init]

    # Set min bound 0 on all coefficients, and set different max bounds for each coefficient
    bounds = (0, [4.])
    # Data
    x = x_train[:-1]
    y = x_train[1:]

    lsq_solution = optim.least_squares(fun, p0, bounds=bounds, loss = 'soft_l1', args=(x, y))
    r = lsq_solution.x[0]

    theta_hat = {'r':r, 'sigma':None}

    return lsq_solution, theta_hat