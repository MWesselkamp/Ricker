import numpy as np

def squared_error_SNR(obs, pred):
    """
    The squared error based SNR, an estimator of the true expected SNR. (Czanner et al. 2015, PNAS)
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Independent of sample size!
    """

    EPE_mean = np.dot(np.transpose(obs - np.mean(obs)), (obs - np.mean(obs))) # returns a scalar
    EPE_model = np.dot(np.transpose(obs - pred), (obs - pred)) # also returns a scalar
    signal = (EPE_mean - EPE_model)
    noise = EPE_model

    return signal/noise


def var_based_SNR(obs, pred, inital_uncertainty):
    """
    The squared error based SNR, an estimator of the true expected SNR at perfect knowledge of parameters.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Dependent on sample size (decreases with sample size)

    # This may be more suited for the perfect model scenario
    # but I am not going to use it until I am sure of what the individual parts are
    """
    signal = np.dot(np.transpose(pred - np.mean(obs)), (pred - np.mean(obs)))
    noise = len(obs) * inital_uncertainty ** 2

    return signal / noise

def raw_SNR(pred, var = False):
    # tSNR raw SNR or timeseries SNR: mean(timeseries) / var(timeseries)
    # tsnr increases with sample size (see sd).

    signal = np.mean(pred, axis=1)
    if var:
        noise = np.mean(pred**2, axis=1) - np.mean(pred, axis=1)**2 # np.std(pred, axis=1)#1/pred.shape[0]*np.sum(np.subtract(pred, mu)**2, axis=0)
    else:
        noise = np.std(pred,axis=1)

    return signal/noise

def raw_CNR(obs, pred):
    """
    CNR - contrast to noise ratio: mean(condition-baseline) / std(baseline)
    This is basically the same as the square-error-based SNR?
    Transfered, we have the model as the baseline and the mean as condition.
    tsnr increases with sample size (see sd).
    """
    signal = np.mean(pred - np.mean(obs)) # returns a scalar
    noise = np.std(obs)

    return signal/noise


def bs_sampling(obs, pred, snr, samples=100):

    its = obs.shape[1]
    arr = np.zeros((its, samples))

    for j in range(samples):

        obs_ind, pred_ind = np.random.randint(obs.shape[0], size=1), np.random.randint(pred.shape[0], size=1)
        x_obs = obs[obs_ind].flatten()
        x_pred = pred[pred_ind].flatten()

        for i in range(its):

            if snr == "cnr":
                arr[i, j] = raw_CNR(x_obs[:i + 2], x_pred[:i + 2])
            elif snr == "ss-snr":
                arr[i, j] = squared_error_SNR(x_obs[:i + 2], x_pred[:i + 2])

    return arr