import numpy as np

def rolling_historic_mean(observations):

    hmean = np.zeros(observations.shape)
    hvar = np.zeros(observations.shape)
    hmean[:,0] = observations[:,0]

    for i in range(1, observations.shape[1]):
        hmean[:,i] = np.mean(observations[0,:i], axis=0)
        hvar[:,i] = np.std(observations[0,:i], axis=0)

    return hmean

def historic_mean(observations):
    hmean = np.mean(observations, axis=0)
    hvar = np.std(observations, axis=0)

    return np.full(observations.shape, hmean)

def persistence(observations):

    x_pred = observations[:,-1]
    return np.full(observations.shape, x_pred)


