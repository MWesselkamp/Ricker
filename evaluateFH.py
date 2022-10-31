import matplotlib.pyplot as plt
import os
import simulations
import forecast_ensemble
import vizualisations
import numpy as np
import json

# Create predictions and observations
def generate_data(phi_preds, metric = "rolling_corrs"):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=2,
                           ensemble_size=10,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-3},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=2,
                    ensemble_size=1,
                    initial_size=(0.98, 0.98))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-3},
                        show = False)[:,:,0]

    imperfect_ensemble = forecast_ensemble.ImperfectEnsemble(ensemble_predictions=xpreds,
                                                           observations=xobs,
                                                            reference="rolling_climatology")
    imperfect_ensemble.verification_settings(metric = metric,
                                       evaluation_style="single")
    imperfect_ensemble.accuracy()
    am = imperfect_ensemble.accuracy_model
    ar = imperfect_ensemble.accuracy_reference

    metadata = {"simulation": {"meta" : sims.meta,
                               "data": xpreds.tolist()},
                "observation": {"meta": obs.meta,
                                "data" : xobs.tolist()},
                "ensemble": {"meta": imperfect_ensemble.meta,
                             "data": {"model": am.tolist(),
                                      "reference" : ar.tolist()}}
                }

    return(am, ar, metadata)

def accuracy_plot(threshold, phi, accuracy_model, accuracy_reference):

    fig = plt.figure()
    ax = fig.add_subplot()
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

# correlations: rho = np.linspace(0.0, 0.99, 50)
rho = np.linspace(0.0, 0.05, 50)
fh_mod_rho = []
fh_ref_rho = []
fh_tot_rho = []
metric = "absolute_differences"

for j in range(len(rho)):
    print("rho is: ", j)
    phi_preds = np.linspace(0.00, 0.0001, 5)
    fh_mod = []
    fh_ref = []
    fh_tot = []

    for i in range(len(phi_preds)):
        accuracy_model, accuracy_reference, metadata = generate_data(phi_preds = phi_preds[i],
                                                                     metric = metric)

        pathname = f"results/fh_evaluation/threshold/{metadata['ensemble']['meta']['metric']}"
        pathname_js = f"{pathname}/{j}{i}meta.json"
        pathname_fig = f"{pathname}/{j}{i}accuracy.png"
        with open(os.path.abspath(pathname_js), 'w') as fp:
            json.dump(metadata, fp)
        accuracy_plot(rho[j], np.round(phi_preds[i],4), accuracy_model, accuracy_reference)

        if metric == "absolute_differences":
            fh_mod.append(np.argmax(accuracy_model > rho[j], axis=1))
            fh_ref.append(np.argmax(accuracy_reference > rho[j], axis=1))
        else:
            fh_mod.append(np.argmax(accuracy_model < rho[j], axis=1))
            fh_ref.append(np.argmax(accuracy_reference < rho[j], axis=1))

    fh_mod_rho.append(np.array(fh_mod))
    fh_ref_rho.append(np.array(fh_ref))
    fh_tot_rho.append(np.array(fh_tot))
fh_mod_rho = np.array(fh_mod_rho)
fh_ref_rho = np.array(fh_ref_rho)
fh_tot_rho = np.array(fh_tot_rho)


fig = plt.figure()
ax = fig.add_subplot()
for i in range(len(phi_preds)):
    ax.plot(rho, fh_mod_rho[:,i,:].squeeze(), color = "lightblue", alpha = 0.6)
ax.plot(rho, np.mean(fh_mod_rho, axis=1), color = "darkblue", alpha = 0.6)
ax.plot(rho, np.mean(np.mean(fh_mod_rho, axis=1), axis=1), color = "yellow")
ax.set_xlabel('Threshold for rho')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}predicted_fh1.png"))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(rho, fh_mod_rho[:,0,:].squeeze(), color = "darkblue", alpha = 0.6)
ax.plot(rho, fh_mod_rho[:,4,:].squeeze(), color = "lightgreen", alpha = 0.6)
ax.plot(rho, np.mean(fh_mod_rho[:,4,:].squeeze(), axis=1), color = "green", alpha = 0.6)
ax.set_xlabel('Threshold for rho')
ax.set_ylabel('Predicted forecast horizon')
fig.show()
fig.savefig(os.path.abspath(f"{pathname}predicted_fh2.png"))