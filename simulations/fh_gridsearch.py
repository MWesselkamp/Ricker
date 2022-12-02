import sys
sys.path.append('../Ricker')
import models
import numpy as np
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import r2_score

def simulate():

    init_s = 0.99
    r = 0.05
    r1= 0.05
    r2= 0.05

    mf_pars = {'lambda': r, 'K': 1}
    mf_errs = {"sigma":0.0001,"phi":0.0, "init_u":1e-4}

    mo_pars = {'lambda1': r1+0.0001, 'K1': 1, 'alpha':1, 'beta':0.00006, 'lambda2': r2, 'K2': 1, 'gamma':1, 'delta':0.00005}
    mo_errs = {"sigma":0.0001,"phi":0.0002, "init_u":1e-4}

    sf_pars = {"iterations":2*52, "initial_size": init_s, "ensemble_size": 25}
    so_pars = {"iterations":2*52, "initial_size": (init_s, init_s), "ensemble_size": 1}

    mf = models.Ricker_Single(set_seed=False)
    mo = models.Ricker_Multi(set_seed=False)

    mf.parameters(mf_pars, mf_errs)
    mo.parameters(mo_pars, mo_errs)

    yf = mf.simulate(sf_pars)['ts']
    yo = mo.simulate(so_pars)['ts'][:,:,0].squeeze()

    return yf, yo

def proficiency(yf, yo):

    window = 3

    d = []
    for j in range(len(yo) - window):
        d.append(pearsonr(yo[j:j + window], yf[j:j + window])[0])

    return np.array(d)

def model_goodness(yf, yo):

    r2 = []
    for i in range(yf.shape[0]):
        r2.append(r2_score(yo, yf[i, :]))
    return np.array(r2)

def horizon(yf, yo, rho = 0.5):

    p = []
    for i in range(yf.shape[0]):

        p.append(proficiency(yf[i,:], yo))

    p = np.array(p)
    p_reached = (np.mean(p, axis=0) < rho)

    if p_reached.sum() == 0:
        fh = len(yo)
    else:
        fh = np.argmax(p_reached)

    return fh

fh_max = []
fh_real = []
for i in range(20):
    yf, yo = simulate()
    r2 = model_goodness(yf, yo)
    fh_max.append(horizon(yf, yf.mean(axis=0)))
    fh_real.append(horizon(yf, yo))

np.round(ttest_ind(fh_max, fh_real)[0], 4)

