import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model

"""
Explore the dynamics of your model.
"""

N=0.8
its = 30
len_r_values = 30
r_values = np.linspace(0, 5, len_r_values)

# Initialize the time-series
timeseries = np.full((its, len_r_values), N, dtype=np.float)

for i in range(len_r_values):
    for j in range(its):
        timeseries[j,i] = model.ricker(timeseries[j-1,i], r_values[i])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylabel("Population size")
    ax.set_xlabel("Time step")
    plt.plot(np.arange(its), timeseries[j,:])
    fig.show()