import matplotlib.pyplot as plt
import os
import simulations
import forecast_ensemble
import vizualisations
import numpy as np
import json
from utils import legend_without_duplicate_labels

# Create predictions and observations
def generate_data(phi_preds = 0.0001, metric = "rolling_CNR", framework = "perfect"):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=4,
                           ensemble_size=20,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-3},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=4,
                    ensemble_size=1,
                    initial_size=(0.98, 0.99))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0004,'initial_uncertainty': 1e-3},
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
    ax.axhline(y=threshold, color="black", linewidth = 0.9, linestyle="--")
    ax.plot(np.transpose(accuracy_model), color="lightblue", alpha=0.6)
    ax.plot(np.transpose(accuracy_reference), color="red", linewidth=1.5)
    ax.set_xlabel('Time steps (weeks)')
    ax.set_ylabel(f"Accuracy: {metadata['ensemble']['meta']['metric']}")
    plt.text(0.2, 0.03, f"Model obs. error: {phi}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    fig.savefig(os.path.abspath(pathname_fig))

def fh_plot(fhmod,pathname_fig):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(fhmod, aspect = "auto", cmap=plt.cm.gray)
    ax.set_xlabel('Time steps (weeks)')
    ax.set_ylabel(f"Ensemble member")
    fig.savefig(os.path.abspath(pathname_fig))

def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

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

def eval_fh(framework = "imperfect", metric = "rolling_corrs", interval = False):

    if metric == "absolute_differences":
        rho = np.linspace(0.0, 0.05, 5)
    elif metric == "rolling_corrs":
        rho = np.linspace(0.0, 0.99, 20)
    elif metric == "rolling_rsquared":
        rho = np.linspace(0.0, 1.0, 5)
    elif metric == "rolling_CNR":
        rho = np.linspace(0.0, 5.0, 20)

    fh_mod_rho = []
    fh_ref_rho = []
    fh_tot_rho = []
    for j in range(len(rho)):
        print("rho is: ", j)
        phi_preds = np.linspace(0.00, 0.0001, 2)
        fh_mod = []
        fh_ref = []
        fh_tot = []

        for i in range(len(phi_preds)):
            xobs, xpreds, accuracy_model, accuracy_reference, metadata = generate_data(phi_preds = phi_preds[i],
                                                                     metric = metric,
                                                                     framework = framework)

            pathname = f"results/fh_evaluation/{framework}/threshold/{metadata['ensemble']['meta']['metric']}"
            pathname_js = f"{pathname}/{j}{i}meta.json"
            pathname_fig1 = f"{pathname}/{j}{i}accuracy.png"
            pathname_fig2 = f"{pathname}/{j}{i}threshold.png"
            with open(os.path.abspath(pathname_js), 'w') as fp:
                json.dump(metadata, fp)
            accuracy_plot(rho[j], np.round(phi_preds[i],4), accuracy_model, accuracy_reference,
                          pathname_fig1, metadata)

            if metric == "absolute_differences":
                fh_mod.append(np.argmax(accuracy_model > rho[j], axis=1))
                fh_ref.append(np.argmax(accuracy_reference > rho[j], axis=1))
                accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                              pathname_fig1, metadata)

            elif metric == "rolling_rsquared":
                fh_mod.append(np.argmax(accuracy_model > rho[j], axis=1))
                fh_ref.append(np.argmax(accuracy_reference > rho[j], axis=1))
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

                accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                              pathname_fig1, metadata)

            elif metric == "rolling_corrs":
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
                accuracy_plot(rho[j], np.round(phi_preds[i], 4), accuracy_model, accuracy_reference,
                              pathname_fig1, metadata)

        fh_mod_rho.append(np.array(fh_mod))
        fh_ref_rho.append(np.array(fh_ref))
        fh_tot_rho.append(np.array(fh_tot))

    if interval:

        fh_mod_rho = np.array(fh_mod_rho)

        fig = plt.figure()
        ax = fig.add_subplot()
        for i in range(len(phi_preds)):
            ax.plot(rho, fh_mod_rho[:, i, :].squeeze(), color="lightblue", alpha=0.6, label="full ensemble")
        ax.plot(rho, np.mean(fh_mod_rho, axis=1), color="darkblue", alpha=0.6, label="ensemble means")
        ax.plot(rho, np.mean(np.mean(fh_mod_rho, axis=1), axis=1), color="yellow", label="overall mean")
        ax.set_xlabel('Threshold for rho')
        ax.set_ylabel('Predicted forecast horizon')
        legend_without_duplicate_labels(ax)
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}predicted_fh1.png"))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(rho, fh_mod_rho[:, 0, :].squeeze(), color="darkblue", alpha=0.6, label="phi = 0.0")
        ax.plot(rho, fh_mod_rho[:, 1, :].squeeze(), color="lightgreen", alpha=0.6, label="phi = 0.0001")
        ax.plot(rho, np.mean(fh_mod_rho[:, 1, :].squeeze(), axis=1), color="green", alpha=0.6)
        ax.set_xlabel('Threshold for rho')
        ax.set_ylabel('Predicted forecast horizon')
        legend_without_duplicate_labels(ax)
        fig.show()
        fig.savefig(os.path.abspath(f"{pathname}predicted_fh2.png"))

        return fh_mod_rho, rho


if __name__=="__main__":

    frameworks = ["imperfect", "perfect"]
    metric = ["rolling_CNR", "rolling_corrs"] #"absolute_differences", "rolling_rsquared", "rolling_CNR",

    for m in metric:
        fhs = []
        for f in frameworks:
            fh_mod_rho, rho = eval_fh(f, m, interval=True)
            fhs.append(fh_mod_rho)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(rho, fhs[0][:, 1, :], color="lightgreen", alpha=0.6)
        ax.plot(rho, fhs[0][:, 1, :].mean(axis=1), color="darkgreen", alpha=0.6, label="imperfect")
        ax.plot(rho, fhs[1][:, 1, :], color="lightblue", alpha=0.6)
        ax.plot(rho, fhs[1][:, 1, :].mean(axis=1), color="blue", alpha=0.6, label="perfect")
        ax.set_xlabel('Threshold for rho')
        ax.set_ylabel('Predicted forecast horizon')
        legend_without_duplicate_labels(ax)
        fig.savefig(os.path.abspath(f"results/fh_evaluation/threshold_based_{metric}.png"))
        fig.show()
