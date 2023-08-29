# create simulator object

from unused_scripts.simulations import Simulator
import matplotlib.pyplot as plt
from utils import legend_without_duplicate_labels
import os

sims = Simulator(model_type="single-species",
                regime="non-chaotic",
                 environment="non-exogeneous",
                timesteps=10,
                ensemble_size=10,
                initial_size=(0.9)) # here we have to give init for both populations
x = sims.simulate()
mod = sims.ricker
derivative = mod.derive(x)

# create simulator object
sims = Simulator(model_type="single-species",
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
sims = Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=200,
                           ensemble_size=1,
                           initial_size=(0.9, 0.9)) # here we have to give init for both populations
x_true = sims.simulate()


# create simulator object
sims = Simulator(model_type="multi-species",
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

obs = Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
obs.hyper_parameters(simulated_years=1,
                    ensemble_size=1,
                    initial_size=(1, 1))
xobs = obs.simulate(pars={'theta': None,'sigma': 0.00009,'phi': 0.0001,'initial_uncertainty': 0},
                    show = False)[:,:,0]
xtrue = obs.ricker.timeseries_true[:,0]
xtrue2 = obs.ricker.timeseries_true[:,1]


sims = Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
sims.hyper_parameters(simulated_years=1,
                           ensemble_size=15,
                           initial_size=1)
xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00007,'phi': 0.00000,'initial_uncertainty': 0},
                           show = False)

#r_sq = []
#for j in range(1,xpreds.shape[1]):
#    r_sq.append([sklearn.metrics.r2_score(xobs[:,:j].transpose(), xpreds[i, :j]) for i in range(xpreds.shape[0])])
#r_sq = np.array(r_sq)

clim = Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
clim.hyper_parameters(simulated_years=50,
                    ensemble_size=15,
                    initial_size=(0.99, 0.99))
climatology = clim.simulate(pars={'theta': None,'sigma': 0.00009,'phi': 0.0001,'initial_uncertainty': 1e-4},
                    show = False)[:,:,0]
climatology_today = climatology[:,(climatology.shape[1]-xpreds.shape[1]):]

pathname = f"../results/fh_evaluation"

import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)


fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology_today.transpose(), color="lightgray", alpha = 0.7, label="climatology")
plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
plt.plot(xobs.transpose(), color="purple", label="observation")
plt.xlabel("Time steps")
plt.ylabel("Rel. population size")
#legend_without_duplicate_labels(ax)
plt.tight_layout()
ax.set_box_aspect(1)
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/climatology-obs_stable.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
#plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="$Y_{f}$")
#plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="Ensemble mean")
plt.plot(xobs.transpose(), color="purple", alpha = 0.7, label="$y_{obs}$")
#plt.plot(xtrue, color="purple", alpha = 0.7, label="$y_{true}$", linestyle="--")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("N")
legend_without_duplicate_labels(ax)
plt.tight_layout()
ax.set_box_aspect(1)
low_y, high_y = ax.get_ylim()
plt.ylim(low_y, high_y)
#plt.axis('off')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/imperfect_model.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
#plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
#plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="$Y_{f}$")
#plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="Ensemble mean")
plt.plot(xobs.transpose(), color="purple", alpha = 0.7, label="$y_{obs}$")
plt.plot(xtrue, color="purple", alpha = 0.7, label="$y_{true}$", linestyle="--")
plt.plot(xtrue2, color="gray", alpha = 0.7, linestyle="--")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("N")
#legend_without_duplicate_labels(ax)
plt.tight_layout()
ax.set_box_aspect(1)
low_y, high_y = ax.get_ylim()
plt.ylim(low_y, high_y)
plt.axis('off')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/observations_only_blank.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
#plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="$Y_{f}$")
#plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="Ensemble mean")
#plt.plot(xobs.transpose(), color="purple", alpha = 0.7, label="$y_{obs}$")
#plt.plot(xtrue, color="purple", alpha = 0.7, label="$y_{true}$", linestyle="--")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("N")
#legend_without_duplicate_labels(ax)
plt.tight_layout()
ax.set_box_aspect(1)
low_y, high_y = ax.get_ylim()
plt.ylim(low_y, high_y)
plt.axis('off')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/ensemble_only_blank.pdf"))


fig = plt.figure()
ax = fig.add_subplot()
#plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="$Y_{f}$")
plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="$\overline{Y_f}$")
#plt.plot(xobs[:,:52].transpose(), color="purple", alpha = 0.7, label="Observation")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("N")
legend_without_duplicate_labels(ax)
plt.ylim(low_y, high_y)
plt.tight_layout()
ax.set_box_aspect(1)
#plt.axis('off')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/perfect_model.pdf"))




## INTRO


sims = Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
sims.hyper_parameters(simulated_years=1,
                           ensemble_size=40,
                           initial_size=1.0)
xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00009,'phi': 0.0000,'initial_uncertainty': 1e-12},
                           show = False)
xpreds = xpreds[:,:20]

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="Forecast")
#plt.fill_between(np.arange(xpreds.transpose().shape[0]), np.quantile(xpreds.transpose(), 0.05, 1), np.quantile(xpreds.transpose(), 0.95, 1), color="lightblue", alpha = 0.7, label="Forecast")
#plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="Ensemble mean")
#plt.plot(xobs[:,:52].transpose(), color="purple", alpha = 0.7, label="Observation")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.ylim((0.999, 1.001))
plt.axis('off')
plt.tight_layout()
fig.show()
#fig.savefig(os.path.abspath(f"{pathname}/general/intro_filled.pdf"))
fig.savefig(os.path.abspath(f"{pathname}/general/intro_trajs.pdf"))


