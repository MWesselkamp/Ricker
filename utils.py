
import numpy as np

def historic_mean(x_test, x_train, length = 'test'):

    if length=='test':
        length = x_test.shape[0]
    historic_mean = np.full((length), np.mean(x_train), dtype=np.float)
    historic_var = np.full((length), np.std(x_train), dtype=np.float)
    return historic_mean, historic_var





