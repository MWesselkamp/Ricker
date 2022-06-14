import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def lyapunov_efh(lyapunovs, precision, dell0 = 0.01):
    return np.multiply.outer(1/lyapunovs,np.log(precision/dell0))

lyapunovs = np.linspace(-1, 1, 50)
precision = np.linspace(0.1, 0.2, 10)
efhs = lyapunov_efh(lyapunovs, precision)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(lyapunovs, efhs )
ax.set_ylabel("Forecast horizon")
ax.set_xlabel("Lyapunov Exponent")
fig.show()