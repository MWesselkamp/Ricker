import math
from scipy.stats import pearsonr
import numpy as np

def historic_mean(x_test, x_train, length = 'test'):

    if length=='test':
        length = x_test.shape[0]
    historic_mean = np.full((length), np.mean(x_train), dtype=np.float)
    historic_var = np.full((length), np.std(x_train), dtype=np.float)
    return historic_mean, historic_var

def rmse(y, y_pred):
    return math.sqrt(np.square(np.subtract(y,y_pred)).mean())

def forecast_rmse(obs, preds, test_index):
    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    :param preds: predicted time series
    :return: time series of rmse
    """

    forecast_error_distributions = []
    for i in range(preds.shape[0]):
        errors = []
        for j in range(preds[:, test_index:].shape[1] - 1):
            errors.append(rmse(obs[test_index:test_index+j+1], preds[i,test_index:test_index+j+1]))
        forecast_error_distributions.append(errors)
    return forecast_error_distributions

def rolling_corrs(obs, preds, test_index, window = 3):
    forecast_corrs = []
    for i in range(preds.shape[0]):
        corrs = []
        for j in range(test_index, preds.shape[1]-window):
            corrs.append(pearsonr(obs[j:j+window], preds[i,j:j+window])[0])
        forecast_corrs.append(corrs)
    return forecast_corrs


