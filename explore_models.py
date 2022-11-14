# create simulator object
import sklearn.metrics

import simulations
import matplotlib.pyplot as plt
from utils import legend_without_duplicate_labels
import os

sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(0.9)) # here we have to give init for both populations
x = sims.simulate()
mod = sims.ricker
derivative = mod.derive(x)

# create simulator object
import simulations
sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(0.9)) # here we have to give init for both populations
x_true = sims.simulate()
mod = sims.ricker
derivative = mod.derive(x_true)


# create simulator object
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=200,
                           ensemble_size=1,
                           initial_size=(0.9, 0.9)) # here we have to give init for both populations
x_true = sims.simulate()


# create simulator object
import simulations
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=200,
                           ensemble_size=1,
                           initial_size=(0.9, 0.9)) # here we have to give init for both populations
x_true = sims.simulate()


## Determine parameters for coexistence of two species
## Based on: May, R.M. Biological populations with non-overlapping generations: Stable points, stable cycles, and chaos. 1974

def D(a11, a12, a21, a22):
    return (a11*a22 - a12*a21)

def A(a11, K2, lambda2, N2_star, a22, K1, lambda1, N1_star):
    return ( (a11*K2/lambda2*N2_star) + (a22*K1/lambda1*N1_star))

D(1, 0.6, 1, 0.5)


## Make intoduction plots

obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
obs.hyper_parameters(simulated_years=2,
                    ensemble_size=1,
                    initial_size=(0.99, 0.99))
xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0003,'initial_uncertainty': 1e-4},
                    show = False)[:,:,0]


sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=15,
                           initial_size=0.99)
xpreds = sims.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-4},
                           show = False)

r_sq = []
for j in range(1,xpreds.shape[1]):
    r_sq.append([sklearn.metrics.r2_score(xobs[:,:j].transpose(), xpreds[i, :j]) for i in range(xpreds.shape[0])])
r_sq = np.array(r_sq)

clim = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
clim.hyper_parameters(simulated_years=10,
                    ensemble_size=15,
                    initial_size=0.99)
climatology = clim.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-4},
                    show = False)
climatology_today = climatology[:,(climatology.shape[1]-xpreds.shape[1]):]

pathname = f"results/fh_evaluation"

import numpy as np
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="Forecast")
plt.plot(xobs[:,:52].transpose(), color="purple", alpha = 0.7, label="Observation")
plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("Population size")
legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/forecast4.pdf"))