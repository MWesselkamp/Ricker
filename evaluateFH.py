import matplotlib.pyplot as plt

import simulations
import forecast_ensemble
import vizualisations
import numpy as np

# Create predictions and observations
def generate_data(phi_obs, sigma_obs):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=2,
                           ensemble_size=10,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': 0.00,'initial_uncertainty': 1e-3},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous", print=False)
    obs.hyper_parameters(simulated_years=2,
                    ensemble_size=1,
                    initial_size=(0.98, 0.98))
    xobs = obs.simulate(pars={'theta': None,'sigma': sigma_obs,'phi': phi_obs,'initial_uncertainty': 1e-3},
                        show = False)[:,:,0]

    return(xobs, xpreds)

def verify_model(xobs, xpreds):
    imperfect_ensemble = forecast_ensemble.ImperfectEnsemble(ensemble_predictions=xpreds,
                                                           observations=xobs,
                                                            reference="rolling_climatology")
    imperfect_ensemble.verification_settings(metric = "rolling_corrs",
                                       evaluation_style="single")
    imperfect_ensemble.accuracy()
    am = imperfect_ensemble.accuracy_model
    ar = imperfect_ensemble.accuracy_reference
    return(am, ar)

rho = np.linspace(0.4, 0.7, 15)
fh_mod_rho = []
fh_ref_rho = []

for j in rho:
    print("rho is: ", j)
    phi_obs = np.linspace(0.0001, 0.0005, 5)
    fh_mod = []
    fh_ref = []

    for i in phi_obs:
        xobs, xpreds = generate_data(phi_obs = i)
        accuracy_model, accuracy_reference = verify_model(xobs, xpreds)
        fh_mod.append(np.argmax(accuracy_model < j, axis=1))
        fh_ref.append(np.argmax(accuracy_reference < j, axis=1))

    fh_mod_rho.append(np.array(fh_mod))
    fh_ref_rho.append(np.array(fh_ref))
fh_mod_rho = np.array(fh_mod_rho)
fh_ref_rho = np.array(fh_ref_rho)

vizualisations.baseplot(np.mean(fh_mod_rho, axis=1), np.mean(fh_ref_rho, axis=1))
vizualisations.baseplot(np.mean(fh_mod_rho, axis=2), fh_ref_rho.squeeze(), transpose=False)
