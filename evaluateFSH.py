import simulations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)

import numpy as np
import CRPS.CRPS as pscore
import os
from utils import legend_without_duplicate_labels, add_identity

clim = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
clim.hyper_parameters(simulated_years=100,
                           ensemble_size=15,
                           initial_size=0.99)
climatology = clim.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.0001,'initial_uncertainty': 1e-4},
                           show = False)

sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=15,
                           initial_size=0.99)
xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.0001,'initial_uncertainty': 1e-4},
                           show = False)

obs = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
obs.hyper_parameters(simulated_years=2,
                    ensemble_size=1,
                    initial_size=0.99)
xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0003,'initial_uncertainty': 1e-4},
                    show = False)

climatology_today = climatology[:,(climatology.shape[1]-xpreds.shape[1]):]
pathname = f"results/fh_evaluation"

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="climatology")
plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
plt.plot(xobs.transpose(), color="purple", label="observation")
plt.xlabel("Time steps")
plt.ylabel("Population size")
legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/climatology-obs.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology[:,:100].transpose(), color="darkgray", alpha = 0.7, label="climatology")
plt.xlabel("Time steps")
plt.ylabel("Population size")
legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="climatology")
plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
plt.xlabel("Time steps")
plt.ylabel("Population size")
legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/climatology.pdf"))

def eval_fsh(metric = "crps"):

    bt_samples = 20
    p_model = []
    p_ref = []
    for j in range(bt_samples):

        emsemble_index = np.random.randint(0, xpreds.shape[0], 1)
        control = xpreds[emsemble_index, :]
        ensemble = np.delete(xpreds, emsemble_index, axis=0)
        list_model=[]
        list_reference = []
        for i in range(1,control.shape[1]):
            if metric == "crps":
                crps,fcrps,acrps = pscore(ensemble[:,i], control[:,i], ensemble.shape[0]).compute()
                list_model.append(crps)
                crps, fcrps, acrps = pscore(climatology_today[:,i], control[:, i], ensemble.shape[0]).compute()
                list_reference.append(crps)
            elif metric == "absolute_differences":
                abs_diff = np.absolute(ensemble[:,i] - control[:,i]).mean()
                list_model.append(abs_diff)
                abs_diff = np.absolute(climatology_today[:, i] - control[:, i]).mean()
                list_reference.append(abs_diff)

        p_model.append(np.array(list_model))
        p_ref.append(np.array(list_reference))

    p_model = np.array(p_model).squeeze()
    p_ref = np.array(p_ref).squeeze()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(0, 0, p_model.shape[1], color="black", linestyles="--")
    plt.plot(p_model.transpose().mean(axis=0), color="lightblue", alpha = 0.7, label="$f^{*}$")
    plt.plot(p_ref.transpose().mean(axis=0), color="darkgray", alpha = 0.7, label="$f^{ref}$")
    plt.plot(abs(p_ref.transpose().mean(axis=0) - p_model.transpose().mean(axis=0)), color="red",
                 label="$d(f^{*}, f^{ref})$")
    plt.xlabel("Time steps")
    plt.ylabel(metric)
    title = "Perfect"
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/perfect/skill/{metric}/{metric}.pdf"))


    list_model=[]
    list_reference = []
    for i in range(1,control.shape[1]):
        if metric == "crps":
            crps,fcrps,acrps = pscore(ensemble[:,i], xobs[:,i], ensemble.shape[0]).compute()
            list_model.append(crps)
            crps, fcrps, acrps = pscore(climatology_today[:,i], xobs[:, i], ensemble.shape[0]).compute()
            list_reference.append(crps)
        elif metric == "absolute_differences":
            abs_diff = np.absolute(ensemble[:, i] - xobs[:, i]).mean()
            list_model.append(abs_diff)
            abs_diff = np.absolute(climatology_today[:, i] - xobs[:, i]).mean()
            list_reference.append(abs_diff)

    ip_model = np.array(list_model)
    ip_ref = np.array(list_reference)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(0, 0, ip_model.shape[0], color="black", linestyles="--")
    plt.plot(ip_model, color="lightblue", label="$f^{*}$")
    plt.plot(ip_ref, color="darkgray",label="$f^{ref}$")
    plt.plot(abs(ip_ref-ip_model), color="red", label="$d(f^{*}, f^{ref})$")
    plt.xlabel("Time steps")
    plt.ylabel(metric)
    title = "Imperfect"
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/imperfect/skill/{metric}/{metric}.pdf"))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(0, 0, ip_model.shape[0], color="black", linestyles="--")
    plt.plot((p_ref.transpose()/p_model.transpose()).mean(axis=1), color="blue", label="perfect")
    plt.plot((ip_ref/ip_model), color="darkgreen", label="imperfect")
    plt.xlabel("Time steps")
    plt.ylabel(f"Skill: {metric}")
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/fsh_{metric}.pdf"))

if __name__=="__main__":

    metrics = ["absolute_differences", "crps"]
    for m in metrics:
        eval_fsh(m)