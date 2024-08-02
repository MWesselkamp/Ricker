import os.path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm

# Generate data for the first normal distribution
mu, sigma = 0, 1  # mean and standard deviation
x = np.linspace(-5, 5, 100)
y = norm.pdf(x, mu, sigma)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ndists = 90

# Plot the first normal distribution
ax.plot(x, np.zeros_like(x), y, label='Distribution 0', zdir='z',
        color = cm.plasma(1 / ndists), alpha=0.9, linewidth=1.2)

# Plot 10 normal distributions with slightly increasing variance along positive y
for i in range(1, ndists):
    sigma_i = sigma + i * 0.035 # Increase variance slightly
    mu_i = mu + i * 0.09 # shift mean slightly
    y_i = norm.pdf(x, mu_i, sigma_i)
    color = cm.plasma(i / ndists)  # Use a gradient from red to yellow
    ax.plot(x, i * np.ones_like(x), y_i, label=f'Distribution {i}', zdir='z',
            color=color, alpha=0.9, linewidth=1.2)

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_axis_off()
# Show the plot
plt.show()
plt.savefig(os.path.join('results', 'titlepage.pdf'), transparent=True)


