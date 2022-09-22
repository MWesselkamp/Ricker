import numpy as np
from pyinform.dist import Dist
from pyinform.shannon import entropy, relative_entropy
import matplotlib.pyplot as plt
import json
import time
import data_handling
from sklearn.neighbors import KernelDensity
import simulations
import utils

# Shannon Entropy
# Is Zero, when there is no more uncertainty, i.e. the probability associated with an outcome becomes 1.

# step one: Calculate the relative frequency of events occurring in a sequence.
# step two: Sum up the relative frequency and the log(relative frequency) of all events.
# Shannon Entropy for discrete distribution
m = np.random.randint(0,50,1000)
d = Dist(50)
for x in m:
    d.tick(x)
print(entropy(d, b = 50))

# Relative Entropy between posterior (p) and prior (q) distributions: Information gained in switching from prior to posterior.
p = Dist([4,1])
q = Dist([1,1])
relative_entropy(p, q)
relative_entropy(q, p)


# So what distributions, if not discrete empirical distributions?
# Lets use a stepwise Kernel Density Estimation.
# I want a density function for every week of the year.


# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(800)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('non-exogeneous', trend=False)
sims.model_type("single-species")
x = sims.simulate()
x_pred, x_clim = np.split(x, 2, axis=1)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.transpose(x_clim), alpha=0.3, color = "blue")
ax.plot(np.transpose(x_pred), alpha=0.3, color= "red")
ax.set_xlabel('Time steps (Generation)', size=14)
ax.set_ylabel('Population size', size=14)
fig.show()

ys = np.concatenate(x_clim)
kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(ys[:, np.newaxis])  # the higher the bandwidth the smoother
# what is the range we use? The min and max ever realized in that periods.
new_ys = np.linspace(x.min(), x.max(), 300)[:, np.newaxis]
dens = kde.score_samples(new_ys)

fig = plt.figure()
ax = fig.add_subplot()
ax.fill(np.exp(new_ys), dens, alpha=0.3)
ax.set_xlabel('Population size', size=14)
ax.set_ylabel('Density', size=14)
fig.show()

fig = plt.figure()
ax = fig.add_subplot()
densies = []
for i in range(50):
    yp = x_pred[:,i+1]
    kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(yp[:, np.newaxis])  # the higher the bandwidth the smoother
    new_yp = np.linspace(x.min(), x.max(), 300)[:, np.newaxis]
    log_dens = kde.score_samples(new_yp)
    densies.append(log_dens)
    ax.plot(new_yp, np.round(np.exp(log_dens),4), alpha=0.3)
fig.show()

def relative_entropy(p, q, integral = True):
    RE = np.sum(p*np.log(p/q)) if integral else p*np.log(p/q)
    return RE

RE = []
for i in range(50):
    RE.append(relative_entropy(densies[i], dens))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(RE)
fig.show()