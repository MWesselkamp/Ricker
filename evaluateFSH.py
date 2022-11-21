import simulations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)

import numpy as np
import CRPS.CRPS as pscore
from scipy.stats import pearsonr
import os
from utils import legend_without_duplicate_labels, add_identity

def generate_data(years = 2, init = 0.99, state = "stable", make_plots=False):
    clim = simulations.Simulator(model_type="single-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    clim.hyper_parameters(simulated_years=100,
                               ensemble_size=25,
                               initial_size=init)
    climatology = clim.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.0001,'initial_uncertainty': 1e-3},
                               show = False)

    sims = simulations.Simulator(model_type="single-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=years,
                               ensemble_size=25,
                               initial_size=init)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.00005,'initial_uncertainty': 1e-5},
                               show = False)

    obs = simulations.Simulator(model_type="multi-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=years,
                        ensemble_size=1,
                        initial_size=(init, init))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-4},
                        show = False)[:,:,0]

    climatology_today = climatology[:,(climatology.shape[1]-xpreds.shape[1]):]
    pathname = f"results/fh_evaluation"

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(climatology_today.transpose(), color="lightgray", alpha = 0.7, label="climatology")
        plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
        plt.plot(xobs.transpose(), color="purple", label="observation")
        plt.xlabel("Time steps")
        plt.ylabel("Population size")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/general/climatology-obs_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(climatology[:,:100].transpose(), color="lightgray", alpha = 0.7, label="climatology")
        plt.xlabel("Time steps")
        plt.ylabel("Population size")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(climatology_today.transpose(), color="lightgray", alpha = 0.7, label="climatology")
        plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
        plt.xlabel("Time steps")
        plt.ylabel("Population size")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/general/climatology_{state}.pdf"))

    return xobs, xpreds, climatology_today

def eval_fsh(metric = "crps", years = 2, state ="stable", make_plots=True):

    pathname = f"results/fh_evaluation"

    if state == "stable":
        init=0.99
    elif state == "instable":
        init =1.0

    xobs, xpreds, climatology_today = generate_data(years = years, init = init, state =state, make_plots=False)

    bt_samples = 20
    p_model = []
    p_ref = []
    for j in range(bt_samples):

        emsemble_index = np.random.randint(0, xpreds.shape[0], 1)
        control = xpreds[emsemble_index, :]
        ensemble = np.delete(xpreds, emsemble_index, axis=0)
        list_model=[]
        list_reference = []

        if metric == "crps":
            for i in range(1, control.shape[1]):
                crps,fcrps,acrps = pscore(ensemble[:,i], control[:,i], ensemble.shape[0]).compute()
                list_model.append(crps)
                crps, fcrps, acrps = pscore(climatology_today[:,i], control[:, i], ensemble.shape[0]).compute()
                list_reference.append(crps)
        elif metric == "absolute_differences":
            for i in range(1, control.shape[1]):
                abs_diff = np.absolute(ensemble[:,i] - control[:,i]).mean()
                list_model.append(abs_diff)
                abs_diff = np.absolute(climatology_today[:, i] - control[:, i]).mean()
                list_reference.append(abs_diff)
        elif metric == "rolling_corrs":
            for i in range(1, control.shape[1]-6):
                corr = np.mean([pearsonr(control[0,i:i+6], ensemble[j,i:i+6])[0] for j in range(ensemble.shape[0])])
                list_model.append(corr)
                corr = np.mean([pearsonr(control[0,i:i+6], climatology_today[j,i:i+6])[0] for j in range(climatology_today.shape[0])])
                list_reference.append(corr)

        p_model.append(np.array(list_model))
        p_ref.append(np.array(list_reference))

    p_model = np.array(p_model).squeeze()
    p_ref = np.array(p_ref).squeeze()

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, p_model.shape[1], color="black", linestyles="--")
        if metric == "rolling_corrs":
            plt.plot(p_model.mean(axis=0), color="lightblue", alpha=0.7, label="$f$")
            plt.plot(p_ref.mean(axis=0), color="darkgray", alpha=0.7, label="$f^{ref}$")
            plt.plot(abs(p_ref.mean(axis=0) - p_model.mean(axis=0)), color="red",
                     label="$d(f, f^{ref})$")
        else:
            plt.plot(p_model.mean(axis=0), color="lightblue", alpha = 0.7, label="$f$")
            plt.plot(p_ref.mean(axis=0), color="darkgray", alpha = 0.7, label="$f^{ref}$")
            plt.plot(abs(p_ref.mean(axis=0) - p_model.mean(axis=0)), color="red",
                         label="$d(f, f^{ref})$")
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        #fig.savefig(os.path.abspath(f"{pathname}/perfect/skill/{metric}/{metric}.pdf"))


    list_model=[]
    list_reference = []
    if metric == "crps":
        for i in range(1, control.shape[1]):
            crps,fcrps,acrps = pscore(ensemble[:,i], xobs[:,i], ensemble.shape[0]).compute()
            list_model.append(crps)
            crps, fcrps, acrps = pscore(climatology_today[:,i], xobs[:, i], ensemble.shape[0]).compute()
            list_reference.append(crps)
    elif metric == "absolute_differences":
        for i in range(1, control.shape[1]):
            abs_diff = np.absolute(ensemble[:, i] - xobs[:, i]).mean()
            list_model.append(abs_diff)
            abs_diff = np.absolute(climatology_today[:, i] - xobs[:, i]).mean()
            list_reference.append(abs_diff)
    elif metric == "rolling_corrs":
        for i in range(1, control.shape[1]-6):
            corr = np.mean([pearsonr(xobs[0,i:i+6], ensemble[j,i:i+6])[0] for j in range(ensemble.shape[0])])
            list_model.append(corr)
            corr = np.mean([pearsonr(xobs[0,i:i+6], climatology_today[j,i:i+6])[0] for j in range(climatology_today.shape[0])])
            list_reference.append(corr)

    ip_model = np.array(list_model)
    ip_ref = np.array(list_reference)

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ip_model.shape[0], color="black", linestyles="--")
        plt.plot(ip_model, color="lightblue", label="$f$")
        plt.plot(ip_ref, color="darkgray",label="$f^{ref}$")
        plt.plot(abs(ip_ref-ip_model), color="red", label="$d(f, f^{ref})$")
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        #fig.savefig(os.path.abspath(f"{pathname}/imperfect/skill/{metric}/{metric}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ip_model.shape[0], color="black", linestyles="--")
        plt.plot((p_ref.transpose()/p_model.transpose()).mean(axis=1), color="blue", label="perfect")
        plt.plot((ip_ref/ip_model), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel(f"Skill: {metric}")
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        #fig.savefig(os.path.abspath(f"{pathname}/fsh_{metric}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, ip_model.shape[0], color="black", linestyles="--")
        plt.plot(np.log((p_model.transpose() / p_ref.transpose())), color="lightblue")
        plt.plot(np.log((p_model.transpose()/p_ref.transpose()).mean(axis=1)), color="blue", label="perfect")
        plt.plot(np.log(ip_model/ip_ref), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel(f"Skill: {metric}")
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        #fig.savefig(os.path.abspath(f"{pathname}/fsh_{metric}_log_{state}.pdf"))

    return p_model, p_ref, ip_model, ip_ref

if __name__=="__main__":

    #metrics = ["crps", "absolute_differences"]
    #for m in metrics:
    #    p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=1, state="instable")
    #   p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=2, state="stable")

    simus = 100
    m = "crps" #, "absolute_differences"
    fhp_l = []
    fhip_l = []
    fhp = []
    fhip= []
    for i in range(simus):
        p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=2, state="stable", make_plots=True)

        fhp_l.append(np.argmax(np.log((p_ref.transpose()/p_model.transpose()).mean(axis=1)) > 0))
        fhip_l.append(np.argmax(np.log((ip_ref.transpose()/ip_model.transpose()).mean(axis=1)) > 0))

        fhp.append(np.argmax(((p_model.transpose()/p_ref.transpose()).mean(axis=1)) > 1))
        fhip.append(np.argmax(((ip_model.transpose()/ip_ref.transpose()).mean(axis=1)) > 1))



