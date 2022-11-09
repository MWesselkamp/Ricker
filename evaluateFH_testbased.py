from scipy.stats import ttest_ind
import simulations
import matplotlib.pyplot as plt
import numpy as np

# Create predictions and observations
def generate_data(years = 2, phi_preds = 0.0001):
    sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    sims.hyper_parameters(simulated_years=years,
                           ensemble_size=10,
                           initial_size=0.99)
    xpreds = sims.simulate(pars={'theta': None,'sigma': 0.00,'phi': phi_preds,'initial_uncertainty': 1e-3},
                           show = False)

    obs = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous", print=False)
    obs.hyper_parameters(simulated_years=years,
                    ensemble_size=1,
                    initial_size=(0.98, 0.99))
    xobs = obs.simulate(pars={'theta': None,'sigma': 0.0001,'phi': 0.0001,'initial_uncertainty': 1e-3},
                        show = False)[:,:,0]

    return xpreds, xobs

xpreds, xobs = generate_data(years = 6)

fig = plt.figure()
plt.plot(np.transpose(xpreds), color="blue")
plt.plot(np.transpose(xobs), color="red")
fig.show()

framework = "perfect"

def calculate_tstats(xobs, xpreds):
    tstats = np.zeros((xpreds.shape[0], xpreds.shape[1]))
    pvalues = np.zeros((xpreds.shape[0], xpreds.shape[1]))
    for j in range(xpreds.shape[0]):
        for i in range(1, xpreds.shape[1]):
                ttest_results = ttest_ind(xobs.flatten(), xpreds[j,:i].flatten(), equal_var=False)
                tstats[j, i] = ttest_results.statistic
                pvalues[j, i] = ttest_results.pvalue
    return tstats, pvalues

if framework == "perfect":
    bt_samples = 50
    emsemble_index = np.random.randint(0, xpreds.shape[0], 1)
    control = xpreds[emsemble_index, :]
    ensemble_n = np.delete(xpreds, emsemble_index, axis=0)
    tstatss = []
    pvaluess = []
    for i in range(bt_samples):
        tstatss.append(calculate_tstats(control, ensemble_n)[0])
        pvaluess.append(calculate_tstats(control, ensemble_n)[1])
    tstats= np.mean(np.array(tstatss), axis=0)
    pvalues = np.mean(np.array(pvaluess), axis=0)
elif framework == "imperfect":
    tstats, pvalues = calculate_tstats(xobs, xpreds)

fig = plt.figure()
plt.hlines(0, 0, tstats.shape[1], color="darkgray", linestyles="--")
plt.plot(np.transpose(tstats), color="blue", alpha=0.5)
plt.ylabel("T-test statistics")
fig.show()

fig = plt.figure()
plt.plot(np.transpose(pvalues), color= "red")
fig.show()


fhs = [np.argmax(tstats[j,2:] < 0) for j in range(tstats.shape[0])]
fhs = [tstats.shape[1] if x == 0 else x for x in fhs]
