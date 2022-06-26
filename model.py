import numpy as np

def ricker(N, r, sigma=None, seed = 100):

    """
    Ricker according to Petchey (2015) or ecolmod R package for a carrying capacity k of 1.
    """

    num = np.random.RandomState(seed)
    if sigma is None:
        return N*np.exp(r*(1-N))
    else:
        return N*np.exp(r*(1-N)) + sigma*num.normal(0, 1)

def ricker_derivative(N, r):
    return np.exp(r-r*N)*(1-r*N)

def iterate_ricker(theta, its, init, obs_error = False, seed = 100):

    """
    Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.

    """
    r = theta['r']
    sigma = theta['sigma']

    num = np.random.RandomState(seed)

    # Initialize the time-series
    timeseries = np.full((its), init, dtype=np.float)
    # Initalize timeseries for lyapunov exponent
    timeseries_log_abs = np.zeros((its), dtype=np.float)
    timeseries_log_abs[0] = np.log(abs(ricker_derivative(init, r)))

    for i in range(1, its):

        timeseries[i] = ricker(timeseries[i-1], r, sigma)
        if obs_error:
            # this can't be a possion distribution! Change this to something else.
            timeseries[i] = num.poisson(timeseries[i])

        timeseries_log_abs[i] = np.log(abs(ricker_derivative(timeseries[i], r)))

    return timeseries, timeseries_log_abs

def ricker_simulate(samples, its, theta, init, obs_error = False, seed=100):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        """
        timeseries_array = [None]*samples
        lyapunovs_array = [None] * samples
        # initialize random number generator
        num = np.random.RandomState(seed)

        for n in range(samples):

            init_sample = num.normal(init[0], init[1])
            while init_sample < 0:
                init_sample = num.normal(init[0], init[1])

            timeseries, timeseries_log_abs = iterate_ricker(theta, its, init_sample, obs_error)
            timeseries_array[n] = timeseries
            lyapunovs_array[n] = timeseries_log_abs

        if samples != 1:
            return np.array(timeseries_array), np.array(lyapunovs_array)
        else:
            return np.array(timeseries_array)[0], np.array(lyapunovs_array)
