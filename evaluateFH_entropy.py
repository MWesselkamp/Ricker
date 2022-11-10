import simulations
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm, entropy
from utils import legend_without_duplicate_labels, add_identity

# Create predictions and observations
def generate_data(years = 2, phi_preds = 0.0001):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=years,
                           ensemble_size=10,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-3},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
    obs.hyper_parameters(simulated_years=years,
                    ensemble_size=1,
                    initial_size=(0.99, 0.99))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-3},
                        show = False)[:,:,0]

    return xpreds, xobs


pathname = f"results/fh_evaluation/"
xpreds, xobs = generate_data(years = 3)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.transpose(xpreds), color="blue", label="forecast")
ax.plot(np.transpose(xobs), color="red", label="observation")
legend_without_duplicate_labels(ax)
fig.show()

fig = plt.figure()
ax = fig.add_subplot()
for i in range(xpreds.shape[0]):
    ax.scatter(np.transpose(xpreds[i,:200]), np.transpose(xobs[:,:200]),
               color="gray", alpha = 0.8, s=20)
add_identity(ax, color='r', ls='--')
ax.set_aspect('equal', adjustable='box')
plt.ylim((0.99, 1.001))
plt.xlim((0.99, 1.001))
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Forecast")
plt.ylabel("Observation")
#legend_without_duplicate_labels(ax)
fig.show()

l = np.linspace(0.01,0.99,20)
qs = np.quantile(xobs.flatten(), l)
hist_clim, bin_edges = np.histogram(xobs, bins = qs, range=(xobs.min(), xobs.max()), density=True)
hist_clim.sum()
probs = hist_clim*np.diff(bin_edges)
fig = plt.figure()
plt.bar(bin_edges[:-1],hist_clim,width=0.0001)
fig.show()

mu, std = norm.fit(xobs.flatten())
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

fig = plt.figure()
plt.hist(xobs.flatten(), bins=20, density=True, alpha=0.6, color='g')
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.4f,  std = %.4f" % (mu, std)
plt.title(title)
fig.show()

ent_ip = []
for i in range(xpreds.shape[1]):
    mu, std = norm.fit(xpreds[:,:i].flatten())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    q = norm.pdf(x, mu, std)
    ent_ip.append(entropy(p, q))

## Perfect

bt_samples = 50
ent_p = []
for j in range(bt_samples):

    emsemble_index = np.random.randint(0, xpreds.shape[0], 1)
    control = xpreds[emsemble_index, :]
    ensemble_n = np.delete(xpreds, emsemble_index, axis=0)

    mu, std = norm.fit(control.flatten())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ent_pj = []
    for i in range(ensemble_n.shape[1]):
        mu, std = norm.fit(ensemble_n[:,:i].flatten())
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        q = norm.pdf(x, mu, std)
        ent_pj.append(entropy(p, q))
    ent_p.append(ent_pj)

ent_p = np.array(ent_p)

fig = plt.figure()
ax = fig.add_subplot()
ax.hlines(0, 0, ent_p.shape[1], color="darkgray", linestyles="--")
plt.plot(np.log(np.transpose(ent_p)), color="lightblue", alpha = 0.7, label="perfect")
plt.plot(np.log(np.array(ent_ip)), color="darkgreen", label="imperfect")
plt.xlabel("Time steps")
plt.ylabel("Log(Relative Entropy)")
legend_without_duplicate_labels(ax)
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/kldivergence.png"))