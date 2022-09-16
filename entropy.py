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
# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=5,
                           ensemble_size=30,
                           initial_size=20)
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('exogeneous', trend=False)
sims.model_type("single-species")
x = sims.simulate()

x_train, x_test = data_handling.split_data(x, t= 52*4)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(0,x_train.shape[1]), np.transpose(x_train), alpha=0.3,
        color = "blue", label="Train set")
ax.plot(np.arange(x_train.shape[1],x.shape[1]), np.transpose(x_test), alpha=0.3,
        color= "red", label="Test set")
ax.legend()
fig.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.hist(np.concatenate(x_train),bins = "auto",  histtype="bar", density=True)
fig.show()

woy = np.concatenate(np.array([np.arange(52)]*int(x_train.shape[1]/52)))

fig, ax = plt.subplots(nrows=5)
for i in range(5):
    # Now we just summarize all weekly data of all ensembles we have into one vector.
    ys = np.concatenate(x_train[:,woy==i])
    ax[i].hist((ys), bins=100, histtype="bar", density=True)
fig.show()

dens_train = np.zeros((52, 100))
kdes = {}
for i in range(52):
    # Now we just summarize all weekly data of all ensembles we have into one vector.
    ys = np.concatenate(x_train[:,woy==i])
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(ys[:,np.newaxis]) # the higher the bandwidth the smoother
    kde_new = 'kde{}'.format(i)
    kdes[kde_new] = kde
    # what is the range we use? The min and max ever realized in that periods.
    new_ys = np.linspace(ys.min(), ys.max(), 100)[:,np.newaxis]
    log_dens = kde.score_samples(new_ys)
    dens_train[i] = np.exp(log_dens)
#with open(f"data/{time.strftime('%Y%m%d-%H%M')}_kdes", 'w') as fp:
#    json.dump(kdes, fp)

# Do we have probabilities?
np.sum(dens_train, axis=1) # No! Then what do we get?

fig = plt.figure()
ax = fig.add_subplot()
for i in range(52):
    ax.fill(np.linspace(x.min(), x.max(), 100), dens_train[i], alpha=0.3)
fig.show()

dens_test = []
for i in range(1):
    kde = kdes[f'kde{i}']
    log_dens_test = kde.score_samples(np.sort(x_test[:,i])[:,np.newaxis])
    log_dens_train = kde.score_samples(np.sort(x_train[:, i])[:, np.newaxis])
    dens_test.append(log_dens)


fig = plt.figure()
ax = fig.add_subplot()
for i in range(1):
    ax.fill(np.sort(x_test[:,i]), np.exp(dens_test[i]), alpha=0.3, color="red")
    ax.fill(np.sort(x_train[:, i]), np.exp(log_dens_train), alpha=0.3, color="blue")
fig.show()
[i for i, x in enumerate(np.round(utils.simulate_T(5*52), 4) ==0) if x]

# Something doesn't seem to work.

# Do the same without weekly.

ys = np.concatenate(x_train)
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(ys[:, np.newaxis])  # the higher the bandwidth the smoother
# what is the range we use? The min and max ever realized in that periods.
new_ys = np.linspace(ys.min(), ys.max(), 100)[:, np.newaxis]
log_dens = kde.score_samples(new_ys)

fig = plt.figure()
ax = fig.add_subplot()
ax.fill(new_ys, np.exp(log_dens), alpha=0.3)
fig.show()

log_dens_test = kde.score_samples(np.sort(x_test[:,0])[:,np.newaxis])

fig = plt.figure()
ax = fig.add_subplot()
ax.fill(np.sort(x_test[:,0]), np.exp(log_dens_test), alpha=0.3)
fig.show()