import numpy as np
import itertools

def ricker(N,r):
    return N*np.exp(r*(1-N))

def iterate_ricker(r, N, its, demo_stoch=False):
    Ns = []
    Ns[0] = N
    for i in range(its):
        if not demo_stoch:
            Ns[i+1] = ricker(N[i], r)
        if demo_stoch:
            expN = ricker(N[i], r)
            Ns[i] = np.random.normal(expN, expN*0.01, 1)
    return np.array(Ns)

if __name__ == '__main__':
    # Distribution from which to choose real value of r
    r_real_mean = 2.9
    r_real_sd = 0
    # Distribution from which to choose real value of N0
    N0_real_mean = 0.8
    N0_real_sd = 0

    # Uncertainty for predictions
    pred_CV = np.array([0, 0.0005, 0.01])
    r_pred_sd = pred_CV*r_real_mean
    N0_pred_sd = pred_CV*N0_real_mean

    demo_stoch = False
    reps = np.arange(10)

