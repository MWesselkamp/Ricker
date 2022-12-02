import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import simulations
from vizualisations import baseplot
# For calculation of permutation entropy
import scipy.stats as ss
from collections import Counter
from math import factorial

# So what distributions, if not discrete empirical distributions?
# Lets use a stepwise Kernel Density Estimation.
sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=15,
                           initial_size=0.99)
x = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.0001,'initial_uncertainty': 1e-4},
                           show = False)

clim = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
clim.hyper_parameters(simulated_years=100,
                           ensemble_size=15,
                           initial_size=0.99)
climatology = clim.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.0001,'initial_uncertainty': 1e-4},
                           show = False)
climatology_today = climatology[:,(climatology.shape[1]-x.shape[1]):]

x_pred, x_clim = np.split(x, 2, axis=1)

baseplot(x_clim, x_pred,transpose=True,
         xlab='Time steps (Generation)',
         ylab='Population size')

ys = np.concatenate(x_clim)
kde = KernelDensity(kernel='gaussian', bandwidth=.8).fit(ys[:, np.newaxis])  # the higher the bandwidth the smoother
# what is the range we use? The min and max ever realized in that periods.
new_ys = np.linspace(x.min(), x.max(), 300)[:, np.newaxis]
dens_clim = np.round(np.exp(kde.score_samples(new_ys)), 50)
dens_clim[dens_clim==0] = 1e-50

fig = plt.figure()
ax = fig.add_subplot()
ax.fill(new_ys, dens_clim, alpha=0.3)
ax.set_xlabel('Population size', size=14)
ax.set_ylabel('Density', size=14)
fig.show()

dens_pred = []
for i in range(x_pred.shape[1]-1): #takes a while, reduce if
    yp = x_pred[:,:i+1].flatten()
    kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(yp[:, np.newaxis])  # the higher the bandwidth the smoother
    new_yp = np.linspace(x.min(), x.max(), 300)[:, np.newaxis]
    log_dens = np.round(kde.score_samples(new_yp), 50)
    log_dens[log_dens == 0] = 1e-50
    dens_pred.append(np.exp(log_dens))


def relative_entropy(p, q, integral = True):
    prob_frac = np.round(p/q, 50)
    prob_frac[prob_frac == 0] = 1e-50
    RE = np.sum(p*np.log(prob_frac)) if integral else p*np.log(prob_frac)
    return RE

def iterate_RE(dens_clim, dens_pred):

    RE = []
    for i in range(x_pred.shape[1]-1): #x_pred.shape[1]-1
        RE.append(relative_entropy(dens_pred[i], dens_clim))
    return RE

    baseplot(RE)

## Running into problems because most densities are 0.
## changed zeros to very small non-zero values. This however, will create a biased RE:
# The probability densities won't actually sum up to 1 anymore.
# One solution: Scale the values
def scale(x):
    return (x-np.mean(x))/np.std(x)

x_clim_scaled = scale(x_clim)
x_pred_scaled  = scale(x_pred)

baseplot(x_clim_scaled, x_pred_scaled,transpose=True,
         xlab='Time steps (Generation)',
         ylab='Population size')

# Let's change approach: Do a histogram over the whole range
# Use quantile bins, self computed.
# We assume the climatological distr doesnt change anymore. Just flatten all available data for the histogram.
l = np.linspace(0.01,0.99,20)
qs = np.quantile(x_clim.flatten(), l)

hist_clim, bin_edges = np.histogram(x_clim, bins = qs, range=(x.min(), x.max()), density=True)
hist_clim.sum()
probs = hist_clim*np.diff(bin_edges)

fig = plt.figure()
plt.bar(bin_edges[:-1],hist_clim,width=1)
fig.show()


# Continued: Let's explore the permutation entropy,
# based on Pennekamp 2019
# PE based on an embedded time series
def embed(x, m, d = 1):
    """
    Pennekamp 2019
    """
    n = len(x) - (m-1)*d
    X = np.arange(len(x))
    out = np.array([X[np.arange(n)]]*m)
    a = np.repeat(np.arange(1, m)*d, out.shape[1])
    out[1:,] = out[1:,]+a.reshape(out[1:,].shape)
    out = x[out]

    return out

def entropy(wd):
    """
    in bits
    """
    return -np.sum(list(wd.values())*np.log2(list(wd.values())))


def word_distr(x_emb, tie_method='average'):

    words = [np.array2string(ss.rankdata(x_emb[:, i])) for i in range(x_emb.shape[1])]
    c = dict(Counter(words))
    for k, v in c.items():
        c[k] = v/len(words)
    return c

def permutation_entropy(x, m, d):

    x_emb = embed(x, m=m)
    wd = word_distr(x_emb)
    denom = np.log2(2 * factorial(m))
    ent = entropy(wd) / denom

    return ent


x_sim = np.random.normal(0,1,30)

words = []
PEs = []
for i in range(x.shape[0]):
    x_emb = embed(x[i,:], m=4)
    wd = word_distr(x_emb)
    words.append(wd)
    denom = np.log2(2*factorial(3))
    ent = entropy(wd)/denom
    PEs.append(ent)

fig = plt.figure()
plt.plot(list(wd.values()))
fig.show()
