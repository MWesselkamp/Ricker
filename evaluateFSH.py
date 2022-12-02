import simulations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)

import numpy as np
import CRPS.CRPS as pscore
import pandas as pd
import os
from utils import legend_without_duplicate_labels, add_identity

def generate_data(years = 2, state = "stable", framework = "perfect", make_plots=False):

    if state == "instable":
        init=0.99
    elif state == "stable":
        init =1.0

    if framework == "perfect":
        clim = simulations.Simulator(model_type="single-species",
                                     simulation_regime="non-chaotic",
                                     environment="non-exogeneous", print=False)
    else:
        clim = simulations.Simulator(model_type="multi-species",
                                     simulation_regime="non-chaotic",
                                     environment="exogeneous", print=False)
    clim.hyper_parameters(simulated_years=50,
                               ensemble_size=25,
                               initial_size=(init, init))
    climatology = clim.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-4},
                               show = False)[:,:,0]

    sims = simulations.Simulator(model_type="single-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=years,
                               ensemble_size=25,
                               initial_size=init)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.000,'initial_uncertainty': 1e-5},
                               show = False)

    obs = simulations.Simulator(model_type="multi-species",
                                 simulation_regime="non-chaotic",
                                 environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=years,
                        ensemble_size=1,
                        initial_size=(init, init))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0002,'initial_uncertainty': 1e-4},
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
        ax.set_box_aspect(1)
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/general/climatology-obs_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(climatology[:,:100].transpose(), color="lightgray", alpha = 0.7, label="climatology")
        plt.xlabel("Time steps")
        plt.ylabel("Population size")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        ax.set_box_aspect(1)
        fig.show()

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(climatology_today.transpose(), color="lightgray", alpha = 0.7, label="climatology")
        plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
        plt.xlabel("Time steps")
        plt.ylabel("Population size")
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        ax.set_box_aspect(1)
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/general/climatology_{state}.pdf"))

    return xobs, xpreds, climatology_today

def eval_fsh(metric, state, years = 2, make_plots=True):

    pathname = f"results/fh_evaluation"

    xobs, xpreds, climatology_today = generate_data(years = years, state =state, framework = "perfect", make_plots=False)

    p_model = []
    p_ref = []

    if metric != "snr":
        # Perfect Model

        bt_samples = 20

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

            p_model.append(np.array(list_model))
            p_ref.append(np.array(list_reference))

        p_model = np.array(p_model).squeeze()
        p_ref = np.array(p_ref).squeeze()

    else:
        for i in range(xpreds.shape[1]):
            p_model.append(xpreds[:,:i].mean()/xpreds[:,:i].std())
            p_ref.append(climatology_today[:, :i].mean() / climatology_today[:, :i].std())
        p_model = np.array(p_model)
        p_ref = np.array(p_ref)

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, xpreds.shape[1], color="black", linestyles="--")
        if metric != "snr":
            plt.plot(p_model.mean(axis=0), color="lightblue", alpha = 0.7, label="$f$")
            plt.plot(p_ref.mean(axis=0), color="darkgray", alpha = 0.7, label="$f^{ref}$")
            plt.plot(abs(p_ref.mean(axis=0) - p_model.mean(axis=0)), color="red",
                             label="$d(f, f^{ref})$")
        else:
            plt.plot(p_model, color="lightblue", alpha=0.7, label="$f$")
            plt.plot(p_ref, color="darkgray", alpha=0.7, label="$f^{ref}$")
            plt.plot(abs(p_ref - p_model), color="red",
                     label="$d(f, f^{ref})$")
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        #fig.savefig(os.path.abspath(f"{pathname}/perfect/skill/{metric}/{metric}.pdf"))

    # Imperfect Model
    xobs2, xpreds2, climatology_today = generate_data(years=years, state=state, framework="imperfect",
                                                    make_plots=False)

    if metric != "snr":
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

        ip_model = np.array(list_model)
        ip_ref = np.array(list_reference)
    else:
        ip_model = []
        ip_ref = []
        for i in range(xpreds.shape[1]):
            ip_model.append(xpreds[:,:i].mean()/xpreds[:,:i].std())
            ip_ref.append(climatology_today[:, :i].mean() / climatology_today[:, :i].std())
        ip_model = np.array(ip_model)
        ip_ref = np.array(ip_ref)

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, xpreds.shape[1], color="black", linestyles="--")
        plt.plot(ip_model, color="lightblue", label="$f$")
        plt.plot(ip_ref, color="darkgray",label="$f^{ref}$")
        plt.plot(abs(ip_ref-ip_model), color="red", label="$d(f, f^{ref})$")
        plt.xlabel("Time steps")
        plt.ylabel(metric)
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/imperfect/skill/{metric}/{metric}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(1, 0, xpreds.shape[1], color="black", linestyles="--")
        if metric == "snr":
            plt.plot((p_model.transpose()/p_ref.transpose()), color="blue", label="perfect")
            plt.plot((ip_model/ip_ref), color="darkgreen", label="imperfect")
        else:
            plt.plot((p_ref.transpose() / p_model.transpose()), color="lightblue", label="perfect")
            plt.plot(((p_ref.transpose()/p_model.transpose())), color="blue", label="perfect")
            plt.plot((ip_ref/ip_model), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel(f"Skill: {metric}")
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/fsh_{metric}_{state}.pdf"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hlines(0, 0, xpreds.shape[1], color="black", linestyles="--")
        if metric == "snr":
            plt.plot(np.log(p_model.transpose() / p_ref.transpose()), color="blue", label="perfect")
            plt.plot(np.log(ip_model / ip_ref), color="darkgreen", label="imperfect")
        else:
            plt.plot(np.log(p_ref.transpose() / p_model.transpose()), color="lightblue", label="perfect")
            plt.plot((np.log(p_ref.transpose() / p_model.transpose())), color="blue", label="perfect")
            plt.plot(np.log(ip_ref / ip_model), color="darkgreen", label="imperfect")
        plt.xlabel("Time steps")
        plt.ylabel(f"Skill: {metric} [Log]")
        legend_without_duplicate_labels(ax)
        ax.set_box_aspect(1)
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}/fsh_{metric}_log_{state}.pdf"))

    return p_model, p_ref, ip_model, ip_ref

if __name__=="__main__":

    #metrics = ["crps", "absolute_differences"]
    #for m in metrics:
    #    p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=1, state="instable")
    #   p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=2, state="stable")

    simus = 100
    ms = ["snr"] #, "absolute_differences"
    states = ["instable", "stable"]
    pathname = f"results/fh_evaluation/"

    for m in ms:
        for s in states:
            fhp_l = []
            fhip_l = []
            fhp = []
            fhip = []
            columns = ['perfect', 'imperfect', 'perfect_l', 'imperfect_l']
            dfs = []
            for i in range(simus):

                p_model, p_ref, ip_model, ip_ref = eval_fsh(m, years=3, state=s, make_plots=False)

                if m =="snr":
                    fhp_l.append(np.argmax(np.log(p_model.transpose()/p_ref.transpose()) < 0))
                    fhip_l.append(np.argmax(np.log(ip_model/ip_ref) < 0))

                    fhp.append(np.argmax((p_model.transpose()/p_ref.transpose()) < 1))
                    fhip.append(np.argmax((ip_model/ip_ref) < 1))
                else:

                    fhp_l.append(np.argmax(np.log(p_ref.transpose()/p_model.transpose()).mean(axis=1) < 0))
                    fhip_l.append(np.argmax(np.log(ip_ref/ip_model) < 0))

                    fhp.append(np.argmax((p_ref.transpose()/p_model.transpose()).mean(axis=1) < 1))
                    fhip.append(np.argmax((ip_ref/ip_model) < 1))

            dfs.append(pd.DataFrame([fhp, fhip, fhp_l, fhip_l], index=columns).T)

            df = pd.concat(dfs)

            df["perfect"] = pd.to_numeric(df["perfect"])
            df["imperfect"] = pd.to_numeric(df["imperfect"])
            df["perfect_l"] = pd.to_numeric(df["perfect_l"])
            df["imperfect_l"] = pd.to_numeric(df["imperfect_l"])

            os.makedirs('results/fh_evaluation', exist_ok=True)
            df.to_csv(f"results/fh_evaluation/fsh_simu_{m}_{s}.csv", index=False)

            fig = plt.figure()
            ax = fig.add_subplot()
            plt.scatter(fhp, fhip)
            low_y, high_y = ax.get_ylim()
            low_x, high_x = ax.get_xlim()
            plt.xlabel("$h_{max}$")
            plt.ylabel("$h_{real}$")
            plt.ylim((low_y, high_x))
            plt.xlim((low_y, high_x))
            plt.tight_layout()
            ax.set_box_aspect(1)
            add_identity(ax, color="r", ls="--")
            fig.show()
            fig.savefig(os.path.abspath(f"{pathname}/fsh_{m}_{s}_reg.pdf"))

            fig = plt.figure()
            ax = fig.add_subplot()
            plt.scatter(fhp_l, fhip_l)
            low_y, high_y = ax.get_ylim()
            low_x, high_x = ax.get_xlim()
            plt.xlabel("$h_{max}$")
            plt.ylabel("$h_{real}$")
            plt.ylim((low_y, high_x))
            plt.xlim((low_y, high_x))
            plt.tight_layout()
            ax.set_box_aspect(1)
            add_identity(ax, color="r", ls="--")
            fig.show()
            fig.savefig(os.path.abspath(f"{pathname}/fsh_{m}_{s}_reg_l.pdf"))
