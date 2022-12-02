import simulations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)
import numpy as np
import os
from scipy.stats import norm, entropy
from utils import legend_without_duplicate_labels, add_identity

# Create predictions and observations
def generate_data(years = 2, init = 0.99, phi_preds = 0.0001, state ="stable",
                  pathname = f"results/fh_evaluation/entropy"):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=years,
                           ensemble_size=25,
                           initial_size=init)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-4},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=years,
                    ensemble_size=1,
                    initial_size=(init, init))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0003,'initial_uncertainty': 1e-4},
                        show = False)[:,:,0]

    clim = simulations.Simulator(model_type="single-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    clim.hyper_parameters(simulated_years=100,
                          ensemble_size=25,
                          initial_size=init)
    climatology = clim.simulate(pars={'theta': None, 'sigma': 0.00, 'phi': 0.0001, 'initial_uncertainty': 1e-3},
                                show=False)
    climatology_today = climatology[:, (climatology.shape[1] - xpreds.shape[1]):]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(climatology_today.transpose(), color="lightgray", alpha=0.7, label="Climatology")
    plt.plot(xpreds.transpose(), color="lightblue", alpha=0.7, label="Forecast")
    plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha=0.7, label="Ensemble mean")
    plt.plot(xobs.transpose(), color="purple", alpha=0.7, linestyle="--")
    # plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
    # plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
    # plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
    plt.xlabel("Time steps")
    plt.ylabel("Population size")
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/data_{state}.pdf"))

    return xpreds, xobs, climatology_today

xpreds, xobs, climatology_today = generate_data(years = 2)
pathname = f"results/fh_evaluation/entropy"

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(climatology_today.transpose(), color="darkgray", alpha = 0.7, label="Climatology")
plt.plot(xpreds.transpose(), color="lightblue", alpha = 0.7, label="Forecast")
plt.plot(xpreds.transpose().mean(axis=1), color="blue", alpha = 0.7, label="Ensemble mean")
plt.plot(xobs.transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(np.arange(52,104),xobs[:,52:].transpose(), color="purple", alpha = 0.7, linestyle="--")
#plt.plot(xpreds[4,:].transpose(), color="blue", alpha = 0.7, linestyle="-")
#plt.vlines(xobs[:,:52].shape[1], 0.99, 1.002, linestyles="-", color="black")
plt.xlabel("Time steps")
plt.ylabel("Population size")
legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/data.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
for i in range(xpreds.shape[0]):
    ax.scatter(np.transpose(xpreds[i,:100]), np.transpose(xobs[:,:100]),
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
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/raw_scatter1.pdf"))

fig = plt.figure()
ax = fig.add_subplot()
for i in range(xpreds.shape[0]):
    ax.scatter(np.transpose(xpreds[i,:]), np.transpose(xobs[:,:]),
               color="gray", alpha = 0.8, s=20)
add_identity(ax, color='r', ls='--')
ax.set_aspect('equal', adjustable='box')
plt.ylim((0.998, 1.002))
plt.xlim((0.998, 1.002))
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Forecast")
plt.ylabel("Observation")
#legend_without_duplicate_labels(ax)
plt.tight_layout()
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/raw_scatter2.pdf"))


l = np.linspace(0.01,0.99,20)
qs = np.quantile(xobs.flatten(), l)
hist_clim, bin_edges = np.histogram(xobs, bins = qs, range=(xobs.min(), xobs.max()), density=True)
hist_clim.sum()
probs = hist_clim*np.diff(bin_edges)
fig = plt.figure()
plt.bar(bin_edges[:-1],hist_clim,width=0.0001)
fig.show()

def eval_entropy(type = "observations", state = "stable", years = 2):

    pathname = f"results/fh_evaluation/entropy"
    if state == "instable":
        xpreds, xobs, climatology_today = generate_data(years=years, init=1.0, state=state)
    else:
        xpreds, xobs, climatology_today = generate_data(years=years, state=state)

    if type == "climatology":
        ref = climatology_today
    elif type == "observations":
        ref = xobs

    mu, std = norm.fit(ref.flatten())
    xmin, xmax = ref.min(), ref.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    fig = plt.figure()
    #ax = fig.add_subplot()
    plt.hist(ref.flatten(), bins=20, density=True, alpha=0.6, color='g')
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.4f,  std = %.4f" % (mu, std)
    plt.title(title)
    #ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/norm_fit_{type}_{state}.pdf"))

    ent_ip = []
    for i in range(xpreds.shape[1]):
        mu, std = norm.fit(xpreds[:,:i].flatten())
        xmin, xmax = xpreds.min(), xpreds.max()
        x = np.linspace(xmin, xmax, 100)
        q = norm.pdf(x, mu, std)
        ent_ip.append(entropy(p, q))

    ## Perfect
    if type == "observations":
        bt_samples = 50
        ent_p = []
        for j in range(bt_samples):

            emsemble_index = np.random.randint(0, xpreds.shape[0], 1)
            control = xpreds[emsemble_index, :]
            ensemble_n = np.delete(xpreds, emsemble_index, axis=0)

            mu, std = norm.fit(control.flatten())
            xmin, xmax = control.min(), control.max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ent_pj = []
            for i in range(ensemble_n.shape[1]):
                mu, std = norm.fit(ensemble_n[:,:i].flatten())
                xmin, xmax = ensemble_n.min(), ensemble_n.max()
                x = np.linspace(xmin, xmax, 100)
                q = norm.pdf(x, mu, std)
                ent_pj.append(entropy(p, q))
            ent_p.append(ent_pj)

        ent_p = np.array(ent_p)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ent_p.shape[1], color="darkgray", linestyles="--")
        plt.plot(np.transpose(ent_p), color="lightblue", alpha = 0.7, label="perfect")
        plt.plot(np.transpose(ent_p).mean(axis=1), color="blue", alpha = 0.7, label="perfect")
        plt.plot(np.array(ent_ip), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel("Log(Relative Entropy)")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/kldivergence_{type}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ent_p.shape[1], color="darkgray", linestyles="--")
        plt.plot(np.log(np.transpose(ent_p)), color="lightblue", alpha = 0.7, label="perfect")
        plt.plot(np.log(np.transpose(ent_p).mean(axis=1)), color="blue", alpha = 0.7, label="perfect")
        plt.plot(np.log(np.array(ent_ip)), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel("$log$(RE)")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/kldivergence_log_{type}_{state}.pdf"))

    else:
        ent_ip = np.array(ent_ip)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ent_ip.shape[0], color="black", linestyles="--")
        ax.hlines(ent_ip.min(axis=0), 0, ent_ip.shape[0], color="darkgray", linestyles="--")
        plt.plot(np.log(np.array(ent_ip)), color="darkblue")
        plt.xlabel("Time steps")
        plt.ylabel("$log$(RE)")
        #legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/kldivergence_log_{type}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ent_ip.shape[0], color="black", linestyles="--")
        ax.hlines(ent_ip.min(axis=0), 0, ent_ip.shape[0], color="darkgray", linestyles="--")
        plt.plot(np.array(ent_ip), color="darkblue")
        plt.xlabel("Time steps")
        plt.ylabel("RE")
        #legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/kldivergence_{type}_{state}.pdf"))

if __name__=="__main__":

    eval_entropy(type = "observations", state="instable", years=1)
    eval_entropy(type = "climatology", state="instable", years=1)