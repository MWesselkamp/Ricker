import numpy as np
import matplotlib.pyplot as plt

from utils import generate_data
from metrics import rolling_corrs, rolling_crps

np.random.seed(42)

#==========================================#
# The Ricker model for population dynamics #
#==========================================#
timesteps = 120
doy_inits = 40
associations = []
crps = []
for doy in np.arange(0,doy_inits):
    xsim, xobs = generate_data(timesteps=timesteps-doy, growth_rate = 0.95,
                           sigma = 0.00, phi = 0.00, initial_uncertainty = 0.00,
                           doy_0=doy, ensemble_size=15)
    #vizualisations.baseplot(xsim, x2=xobs[:,:,0], transpose=True)
    xsim_leads = xsim[:,(doy_inits-doy):]
    xobs_leads = xobs[:,(doy_inits-doy):,0]
    association = rolling_corrs(xobs_leads, xsim_leads, window=5)
    crps.append(rolling_crps(xobs_leads, xsim_leads)[:,0])
    associations.append(np.mean(association, axis=0))
association_matrix = np.array(associations)
crps_matrix = np.array(crps)


plt.rcParams.update({'font.size': 16})
fig, (ax1, ax2) = plt.subplots(figsize = (14,4), ncols=2)
pos1 = ax1.matshow(association_matrix, cmap='Greys_r')
ax1.matshow(association_matrix, cmap='Greys_r')
ax1.set_ylabel('Initial Day of Year')
ax1.set_xlabel('Lead Day of Year')
ax1.set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(80,160, step=10))
plt.colorbar(pos1, ax= ax1, label = 'Association')
pos2 = ax2.matshow(crps_matrix, cmap='Greys')
ax2.matshow(crps_matrix, cmap='Greys')
ax2.set_ylabel('Initial Day of Year')
ax2.set_xlabel('Lead Day of Year')
ax2.set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(80,160, step=10))
plt.colorbar(pos2, ax=ax2, label = 'CRPS')
fig.tight_layout()
plt.show()
