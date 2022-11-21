import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rc('font', family='sans-serif', size=14)
import os
import simulations
import forecast_ensemble
import pandas as pd
import numpy as np
import json
from utils import legend_without_duplicate_labels

# Create predictions and observations
def generate_data(phi_preds = 0.0001, metric = "rolling_CNR", framework = "perfect"):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=2,
                           ensemble_size=25,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-5},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=2,
                    ensemble_size=1,
                    initial_size=(0.99, 0.99))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-4},
                        show = False)[:,:,0]

    if framework == "perfect":
        ensemble = forecast_ensemble.PerfectEnsemble(ensemble_predictions=xpreds,
                                                    reference="rolling_climatology")
        ensemble.verification_settings(metric=metric,
                                        evaluation_style="single")
        ensemble.accuracy()
        am = ensemble.accuracy_model
        #am = np.mean(am, axis=2)
        ar = ensemble.accuracy_reference
        #ar = np.mean(ar, axis=2)

    else:
        ensemble = forecast_ensemble.ImperfectEnsemble(ensemble_predictions=xpreds,
                                                        observations=xobs,
                                                        reference="rolling_climatology")
        ensemble.verification_settings(metric = metric,
                                       evaluation_style="single")
        ensemble.accuracy()
        am = ensemble.accuracy_model
        ar = ensemble.accuracy_reference

    metadata = {"simulation": {"meta" : sims.meta,
                               "data": xpreds.tolist()},
                "observation": {"meta": obs.meta,
                                "data" : xobs.tolist()},
                "ensemble": {"meta": ensemble.meta,
                             "data": {"model": am.tolist(),
                                      "reference" : ar.tolist()}}
                }

    return(xobs, xpreds, am, ar, metadata)

def data_plot(xobs, xpreds, fh, pathname):

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(xpreds.transpose(), color="lightblue", label="forecast")
    plt.plot(xobs.transpose(), color="purple", label="observation")
    low_y, high_y = ax.get_ylim()
    plt.vlines(fh, low_y, high_y, color="darkgray", linestyles="--")
    plt.xlabel("Time steps")
    plt.ylabel("Population size")
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    fig.savefig(os.path.abspath(pathname))

def accuracy_plot(threshold, phi, accuracy_model, accuracy_reference,
                  pathname_fig, metadata,
                  log = False):

    fig = plt.figure()
    ax = fig.add_subplot()
    if log:
        y = abs(np.nanmin(accuracy_model[accuracy_model != -np.inf]))
        threshold = np.log(threshold+y)
        accuracy_model = np.log(accuracy_model+y)
        accuracy_reference = np.log(accuracy_reference+y)
    ax.plot(np.transpose(accuracy_model), color="lightblue", alpha=0.6)
    ax.plot(np.transpose(accuracy_model).mean(axis=1), color="blue", linewidth=1.0)
    ax.axhline(y=threshold, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel('Time steps (weeks)')
    ax.set_ylabel(f"Accuracy: {metadata['ensemble']['meta']['metric']}")
    plt.text(0.2, 0.03, f"$\phi$: {phi}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(os.path.abspath(pathname_fig))

def fh_plot(fhmod,pathname_fig):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(fhmod, aspect = "auto", cmap=plt.cm.gray)
    ax.set_xlabel('Time steps (weeks)')
    ax.set_ylabel(f"Ensemble member")
    plt.tight_layout()
    fig.savefig(os.path.abspath(pathname_fig))

def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return [Na]

#=================#
# Threshold based #
#=================#

def eval_fh(framework = "imperfect", metric = "rolling_corrs", interval = True, save_metadata=False, make_plots = False):

    if metric == "absolute_differences":
        rho = np.linspace(0.0, 0.005, 50)
    elif metric == "rolling_corrs":
        rho = np.linspace(0.0, 0.99, 50)
    elif metric == "rolling_rsquared":
        rho = np.linspace(0.0, 1.0, 50)
    elif metric == "rolling_CNR":
        rho = np.linspace(0.0, 1.0, 50)

    rho = np.round(rho,2)
    phi_preds = np.linspace(0.00, 0.00005, 2)

    fh_mod_rho = []
    fh_ref_rho = []
    fh_tot_rho = []

    for i in range(len(phi_preds)):
        xobs, xpreds, accuracy_model, accuracy_reference, metadata = generate_data(phi_preds = phi_preds[i],
                                                                 metric = metric,
                                                                 framework = framework)

        fh_mod = []
        fh_ref = []
        fh_tot = []

        for j in range(len(rho)):

            pathname = f"results/fh_evaluation/{framework}/threshold/{metadata['ensemble']['meta']['metric']}"
            pathname_js = f"{pathname}/{j}{i}meta.json"
            pathname_fig1 = f"{pathname}/{j}{i}accuracy.pdf"
            pathname_fig2 = f"{pathname}/{j}{i}threshold.pdf"
            if save_metadata:
                with open(os.path.abspath(pathname_js), 'w') as fp:
                    json.dump(metadata, fp)

            if metric == "absolute_differences":
                if interval:
                    fhmod = (accuracy_model > rho[j])
                    fh_plot(fhmod, pathname_fig2)
                    fhmod_c = np.argmax(np.mean(accuracy_model, axis=1) > rho[j])
                    #fhmod = (accuracy_model.mean(axis=0) > rho[j])
                    # print(search_sequence_numpy(fhmod[i, :], np.array([True, True, True])))
                    fhs = [min(search_sequence_numpy(fhmod[i, :], np.array([True, True, True]))) for i in
                           range(fhmod.shape[0])]
                    fh_mod.append(fhs)
                    if make_plots:
                        data_plot(xobs, xpreds, np.mean(fhmod_c), pathname=f"{pathname}/{j}{i}data_conservative.pdf")
                        data_plot(xobs, xpreds, np.mean(fhs), pathname=f"{pathname}/{j}{i}data_loose.pdf")
                else:
                    fh_mod.append(np.argmax(accuracy_model > rho[j], axis=1))
                    fh_ref.append(np.argmax(accuracy_reference > rho[j], axis=1))
                if make_plots:
                    accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                                  pathname_fig1, metadata)


            elif metric == "rolling_rsquared":
                if interval:
                    fhmod = (accuracy_model < rho[j])
                    fh_plot(fhmod, pathname_fig2)
                    # print(search_sequence_numpy(fhmod[i, :], np.array([True, True, True])))
                    fhs = [min(search_sequence_numpy(fhmod[i, :], np.array([True, True, True]))) for i in
                           range(fhmod.shape[0])]
                    fh_mod.append(fhs)
                else:
                    fh_mod.append(np.argmax(accuracy_model > rho[j], axis=1))
                    fh_ref.append(np.argmax(accuracy_reference > rho[j], axis=1))
                if make_plots:
                    accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                                    pathname_fig1, metadata,
                                    log=False)

            elif metric == "rolling_CNR":
                if interval:
                    fhmod = (accuracy_model < rho[j])
                    fh_plot(fhmod, pathname_fig2)
                    #print(search_sequence_numpy(fhmod[i, :], np.array([True, True, True])))
                    fhs = [min(search_sequence_numpy(fhmod[i, :], np.array([True, True, True]))) for i in
                           range(fhmod.shape[0])]
                    fh_mod.append(fhs)
                else:
                    fh_mod.append(np.argmax(accuracy_model < rho[j], axis=1))
                    fh_ref.append(np.argmax(accuracy_reference < rho[j], axis=1))
                if make_plots:
                    accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                                  pathname_fig1, metadata)

            elif metric == "rolling_corrs":
                if interval:
                    fhmod = (accuracy_model < rho[j])
                    fh_plot(fhmod, pathname_fig2)
                    fhmod_c = np.argmax(np.mean(accuracy_model, axis=0) < rho[j])
                    fhs = [min(search_sequence_numpy(fhmod[i, :], np.array([True, True, True]))) for i in
                           range(fhmod.shape[0])]
                    fh_mod.append(fhs)
                    if make_plots:
                        data_plot(xobs, xpreds, fhmod_c, pathname=f"{pathname}/{j}{i}data_conservative.pdf")
                        data_plot(xobs, xpreds, np.mean(fhs), pathname=f"{pathname}/{j}{i}data_loose.pdf")
                else:
                    fh_mod.append(np.argmax(accuracy_model < rho[j], axis=1))
                    fh_ref.append(np.argmax(accuracy_reference < rho[j], axis=1))
                if make_plots:
                    accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                                  pathname_fig1, metadata)

        fh_mod_rho.append(np.array(fh_mod))
        fh_ref_rho.append(np.array(fh_ref))
        fh_tot_rho.append(np.array(fh_tot))

    fh_mod_rho = np.array(fh_mod_rho)

    if (interval & make_plots) :

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(rho, fh_mod_rho[0, :, :].squeeze(), color="darkblue", alpha=0.6, label="$\phi$ = 0.0")
        ax.plot(rho, fh_mod_rho[1, :, :].squeeze(), color="lightgreen", alpha=0.6, label="$\phi$ = 0.0001")
        ax.plot(rho, np.mean(fh_mod_rho[1, :, :].squeeze(), axis=1), color="green", alpha=0.6)
        ax.set_xlabel("Threshold for rho")
        ax.set_ylabel('Predicted forecast horizon')
        legend_without_duplicate_labels(ax)
        plt.tight_layout()
        fig.savefig(os.path.abspath(f"{pathname}predicted_fh2.pdf"))

    return fh_mod_rho, rho

def run_evaluation(metric = ["rolling_corrs", "absolute_differences"], frameworks = ["imperfect", "perfect"], make_plots = False):

    for m in metric:
        fhs = []
        for f in frameworks:
            print("Framework:", f)
            print("Metric:", m)
            fh_mod_rho, rho = eval_fh(f, m, interval=True, make_plots = make_plots)
            fhs.append(fh_mod_rho)

        if make_plots:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.fill_between(rho, np.quantile(fhs[0][1, :, :], 0.25, axis=1), np.quantile(fhs[0][1, :, :], 0.75, axis=1),
                            color="lightgreen", alpha=0.5)
            ax.plot(rho, np.quantile(fhs[0][1, :, :], 0.5, axis=1), color="darkgreen", alpha=0.7, label="imperfect")
            ax.fill_between(rho, np.quantile(fhs[1][1, :, :], 0.25, axis=1), np.quantile(fhs[1][1, :, :], 0.75, axis=1),
                            color="lightblue", alpha=0.5)
            ax.plot(rho, np.quantile(fhs[1][1, :, :], 0.5, axis=1), color="blue", alpha=0.7, label="perfect")
            ax.set_xlabel('Threshold for Rho')
            ax.set_ylabel('Predicted forecast horizon')
            legend_without_duplicate_labels(ax)
            ax.set_box_aspect(1)
            plt.tight_layout()
            fig.savefig(os.path.abspath(f"results/fh_evaluation/threshold_based_{m}.pdf"))
            fig.show()

def simulate(ms=["rolling_corrs"], nsimus = 100):

    for m in ms:

        frameworks = ["perfect", "imperfect"]
        columns = ["framework", "simulation", "rho", "fh_mean", "fh_std"]
        dfs = []

        for f in frameworks:
            for i in range(nsimus):

                fh_mod_rho, rho = eval_fh(f,m,interval=True, make_plots=False)
                fh_mod_rho = fh_mod_rho[1, :, :]

                fh_mean = np.round(fh_mod_rho.mean(axis=1), 2)
                fh_std = np.round(fh_mod_rho.std(axis=1), 2)
                simu = np.repeat(i, len(rho))
                if f == "perfect":
                    frame = np.repeat(0, len(rho))
                else:
                    frame = np.repeat(1, len(rho))

                dfs.append(pd.DataFrame([frame, simu, rho, fh_mean, fh_std], index=columns).T)

        df = pd.concat(dfs)

        df["simulation"] = pd.to_numeric(df["simulation"])
        df["rho"] = pd.to_numeric(df["rho"])
        df["fh_mean"] = pd.to_numeric(df["fh_mean"])
        df["fh_std"] = pd.to_numeric(df["fh_std"])
        df["framework"] = df["framework"].astype("int")

        os.makedirs('results/fh_evaluation', exist_ok=True)
        df.to_csv(f"results/fh_evaluation/fh_simu_{m}.csv", index=False)

        print(m, "DONE")

    return df

def plot_results(m):

    df = pd.read_csv(f"results/fh_evaluation/fh_simu_{m}.csv")
    df_p = df[df['framework'] == 0]
    df_ip = df[df['framework'] == 1]

    x = np.array(df_p["fh_mean"])
    y = np.array(df_ip["fh_mean"])
    rho = np.array(df_ip["rho"])

    l = np.linspace(0.01, 0.99, 20)
    qs = np.quantile(x.flatten(), l)
    qs_y = np.quantile(y.flatten(), l)
    hx, bx = np.histogram(x, bins=qs, range=(x.min(), x.max()), density=True)
    hy, by = np.histogram(y, bins=qs_y, range=(y.min(), y.max()), density=True)

    pathname = f"results/fh_evaluation/"

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.bar(bx[:-1], hx, color="lightblue", alpha=0.8, width=1.1, label="$h_{max}$")
    plt.bar(by[:-1], hy, color="lightgreen", alpha=0.8, width=1.1, label="$h_{real}$")
    low_y, high_y = ax.get_ylim()
    plt.vlines(x.mean(), low_y, high_y, linestyles="-", color="blue")
    plt.vlines(y.mean(), low_y, high_y, linestyles="-", color="green")
    plt.xlabel("$h$")
    plt.ylabel("Density estimate")
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    ax.set_box_aspect(1)
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/threshold_rolling_corrs_hist.pdf"))

    fig = plt.figure()
    ax = fig.add_subplot()
    p1 = plt.scatter(x, y, c=rho, cmap="Blues", alpha=0.6, label="Rho")
    low_y, high_y = ax.get_ylim()
    low_x, high_x = ax.get_xlim()
    plt.xlabel("$h_{max}$")
    plt.ylabel("$h_{real}$")
    plt.ylim((low_y, high_x))
    plt.xlim((low_y, high_x))
    # legend_without_duplicate_labels(ax)
    fig.colorbar(p1, label='Rho')
    plt.tight_layout()
    ax.set_box_aspect(1)
    fig.show()
    fig.savefig(os.path.abspath(f"{pathname}/threshold_rolling_corrs_reg.pdf"))


if __name__=="__main__":

    run_evaluation(make_plots=True)
    # df = simulate()
    # plot_results()

