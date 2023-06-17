from scipy.stats import ttest_ind
from simulations import Simulator
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import legend_without_duplicate_labels, generate_data

def calculate_tstats(xobs, xpreds):
    tstats = np.zeros((xpreds.shape[0], xpreds.shape[1]))
    pvalues = np.zeros((xpreds.shape[0], xpreds.shape[1]))
    for j in range(xpreds.shape[0]):
        for i in range(1, xpreds.shape[1]):
                ttest_results = ttest_ind(xobs.flatten(), xpreds[j,:i].flatten(), equal_var=False)
                tstats[j, i] = ttest_results.statistic
                pvalues[j, i] = ttest_results.pvalue
    return tstats, pvalues

xpreds, xobs = generate_data(timesteps= 500)

frameworks = ["imperfect", "perfect"]
pathname = f"results/fh_evaluation/"

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.transpose(xpreds), color="blue", label="forecast")
ax.plot(np.transpose(xobs), color="red", label="observation")
legend_without_duplicate_labels(ax)
fig.show()
fig.savefig(os.path.abspath(f"{pathname}/general/dynamics.png"))

fig = plt.figure()
ax = fig.add_subplot()

fhs = []
for f in frameworks:

    if f == "perfect":
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

        ax.hlines(0, 0, tstats.shape[1], color="darkgray", linestyles="--")
        ax.plot(np.transpose(tstats), color="blue", alpha=0.5, label = f)

    elif f == "imperfect":
        tstats, pvalues = calculate_tstats(xobs, xpreds)
        ax.plot(np.transpose(tstats), color="darkgreen", alpha=0.5, label = f)

    fh = [np.argmax(tstats[j, 2:] < 0) for j in range(tstats.shape[0])]
    fh = [tstats.shape[1] if x == 0 else x for x in fh]
    fhs.append(fh)
    print(fh)

fhp = np.round(np.mean(fhs[0]),2)
fhip = np.round(np.mean(fhs[1]),2)

plt.ylabel("T-test statistics")
plt.xlabel("Time steps (weeks)")
plt.text(0.3, 0.89, f"FH imperfect: {fhp}\n FH perfect: {fhip}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
legend_without_duplicate_labels(ax)
fig.show()
fig.savefig(os.path.abspath(f"{pathname}ttest.png"))

#fig = plt.figure()
#plt.plot(np.transpose(pvalues), color= "red")
#fig.show()



