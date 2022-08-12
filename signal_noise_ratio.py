import numpy as np
import models

uncertainties = {"parameters":False, "initial":True,"observation":False,"stoch":False}
# Set hyperparameters.
hp_r = {"iterations":30, "initial_size": 0.8, "initial_uncertainty": 1e-4, "ensemble_size": 50}
# Set parameters
theta_r2 = {'lambda':np.exp(2.9), 'alpha':1, 'sigma':None} # lambda = exp(r)

# Initialize model 2
ricker = models.Ricker_2(uncertainties)
ricker.set_parameters(theta_r2)
simu = ricker.simulate(hp_r)
x = simu["ts"]
ricker.visualise(np.transpose(x))

#===========================================#
# Forecast horizon with signal noise ratio. #
#===========================================#

# tSNR raw SNR or timeseries SNR: mean(timeseries) / std(timeseries)
# CNR - contrast to noise ratio: mean(condition-baseline) / std(baseline)
def tsnr(x):
    """
    timeseries signal-to-noise ratio
    tsnr increases with sample size (see sd).
    """
    mu = np.mean(x)
    sd = 1/x.shape[0]*np.sum(np.subtract(x, mu)**2, axis=0)
    return mu/sd

tsnr(np.transpose(x))


def cnr(x_c, x_b):
    """
    time series contrast-to-noise ratio
    tsnr increases with sample size (see sd).
    """
    mu = np.mean(x_c) - np.mean(x_b)
    sd = 1/x.shape[0]*np.sum(np.subtract(x_b, mu)**2, axis=0)
    return mu/sd

