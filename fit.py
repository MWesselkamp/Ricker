import numpy as np
import scipy.optimize as optim

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
        mod.theta['r'] = pars[0]
        res = mod.model(x) - y
        return res

    # Initialize parameters randomly
    r_init = np.random.normal(mod.theta['r'], mod.theta['r']*mod.precision, 1)[0]
    p0 = [r_init]

    # Data
    x = x_train[:-1]
    y = x_train[1:]

    lsq_solution = optim.least_squares(fun, p0, bounds=bounds, loss = 'soft_l1', args=(x, y))
    r = lsq_solution.x[0]

    mod.theta_hat = {'r':r, 'sigma':None}

    return lsq_solution