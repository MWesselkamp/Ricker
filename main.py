import numpy as np
import matplotlib.pyplot as plt
import visualisations

from utils import generate_data
from references import diurnal_climatology
from metrics import rolling_corrs, rolling_crps

np.random.seed(42)

def metric_map(growth_rates, timesteps, doy_inits, metric):

    if metric == 'deterministic':
        s = 0.00
    elif metric == 'stochastic':
        s = 0.001

    metric_map = []
    for r in growth_rates:
        qualities = []
        for doy in np.arange(0, doy_inits):

            xsim, xobs = generate_data(timesteps=timesteps, growth_rate=r,
                                       sigma=s, phi=0.00, initial_uncertainty=0.00,
                                       doy_0=doy, ensemble_size=15)
            xsim_leads = xsim[:, (doy_inits - doy):(timesteps)]
            xobs_leads = xobs[:, (doy_inits - doy):(timesteps), 0]
            if metric == 'deterministic':
                quality = np.mean(rolling_corrs(xobs_leads, xsim_leads), axis=0)
            elif metric == 'stochastic':
                quality = rolling_crps(xobs_leads, xsim_leads)
            qualities.append(quality)
        qualities_matrix = np.array(qualities)

        # visualisations.baseplot(xsim, x2=xobs[:,:,0], transpose=True)
        # visualisations.baseplot(association,np.mean(association, axis=0), transpose=True)
        # visualisations.baseplot(quality, transpose=False)

        metric_map.append(qualities_matrix)

    return metric_map

def get_climatology(timesteps, doy_inits, metric):

    if metric == 'deterministic':
        s = 0.00
    elif metric == 'stochastic':
        s = 0.001

    clim = diurnal_climatology(growth_rate=0.1, sigma=s, initial_uncertainty=0.00, add_trend=True,
                               add_noise=False)  # create climatology
    clim_subset = clim[:, doy_inits:timesteps, :]

    if metric == 'deterministic':

        clims = clim_subset.mean(axis=0)
        clims = clims.mean(axis=1)
        clims_std = clims.std(axis=1)

    else:
        clims = np.transpose(np.array([clim_subset[:, i, :].flatten() for i in range(80)]))

    return clims

#============================#
# Practical Forecast horizon #
#=============================#

growth_rates = [0.1, 0.95]
timesteps = 120
doy_inits = 40

association = metric_map(growth_rates, timesteps, doy_inits, metric='deterministic')
crps = metric_map(growth_rates, timesteps, doy_inits, metric='stochastic')

plt.rcParams.update({'font.size': 16})
fig, (ax1, ax3) = plt.subplots(figsize = (14,8), ncols=2, nrows=2)
pos1 = ax1[0,].imshow(association[0], cmap='Greys_r')
ax1[0,].matshow(association[0], cmap='Greys_r')
ax1[0,].set_ylabel('Day from Lead')
ax1[0,].set_xlabel('Day of Year')
ax1[0,].set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(doy_inits,timesteps, step=10))
plt.colorbar(pos1, ax= ax1[0,], label = 'Association')
pos2 = ax1[1,].imshow(association[1], cmap='Greys_r')
ax1[1,].matshow(association[1], cmap='Greys_r')
ax1[1,].set_ylabel('Initial Day of Year')
ax1[1,].set_xlabel('Lead Day of Year')
ax1[1,].set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(doy_inits,timesteps, step=10))
plt.colorbar(pos2, ax= ax1[1,], label = 'Association')
pos3 = ax3[0,].matshow(crps[0], cmap='Greys')
ax3[0,].matshow(crps[0], cmap='Greys')
ax3[0,].set_ylabel('Day from Lead')
ax3[0,].set_xlabel('Day of Year')
ax3[0,].set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(doy_inits,timesteps, step=10))
plt.colorbar(pos3, ax=ax3[0,], label = 'CRPS')
pos3 = ax3[1,].matshow(crps[1], cmap='Greys')
ax3[1,].matshow(crps[1], cmap='Greys')
ax3[1,].set_ylabel('Day from Lead')
ax3[1,].set_xlabel('Day of Year')
ax3[1,].set_xticks(ticks=np.arange(0,80, step=10), labels=np.arange(doy_inits,timesteps, step=10))
plt.colorbar(pos3, ax=ax3[1,], label = 'CRPS')
fig.tight_layout()
plt.show()
fig.savefig('results/main/absolute.pdf')

#========================#
# Forecast skill horizon #
#========================#

climatology = get_climatology(timesteps, doy_inits, metric='deterministic')

#============================#
# Intrinsic Forecast horizon #
#============================#




# benchmark against climatology.
clim = diurnal_climatology(growth_rate=0.1, sigma=0.0, initial_uncertainty=0.00, add_trend=True, add_noise=False) # create climatology
xsim, xobs = generate_data(timesteps=365, growth_rate=0.1,
                            sigma=0.00, phi=0.00, initial_uncertainty=0.00,
                            doy_0=0, ensemble_size=15,
                           environment='exogeneous', add_trend=False, add_noise=False) # create simulations
xobs = xobs[:, :, 0] # observations


clims = clim.mean(axis=0)
clim_mean = clims.mean(axis=1)
clim_std = clims.std(axis=1)

plt.plot(np.transpose(clim_mean), color = 'gray', label='climatology')
plt.plot(clim_mean+2*clim_std, color = 'gray', linestyle='--')
plt.plot(clim_mean-2*clim_std, color = 'gray', linestyle='--')
plt.plot(np.transpose(xsim), color = 'blue')
plt.plot(np.transpose(xobs), color = 'red', label='observations')
plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("Relative stock biomass")
plt.savefig('results/main/forecast_skill_1.pdf')

xsim_mean = np.mean(xsim, axis=0)
xsim_abs = abs(xsim_mean - xobs)
clim_abs = abs(clim[0] - xobs)
plt.plot(np.transpose(xsim_abs), color = 'blue')
plt.plot(np.transpose(clim_abs), color = 'gray', label='climatology')
plt.plot(np.subtract(np.transpose(xsim_abs),np.transpose(clim_abs)), color = 'red')

#======================#
# deterministic metric #
#======================#

sigmas = [0.00, 0.01]
growth_rates = [0.1, 0.95]
timesteps = 120
doy_inits = 40
skills_sigma = []
metric = 'stochastic'
for s in sigmas:

    clim = diurnal_climatology(growth_rate=0.1, sigma=s, initial_uncertainty=0.00, add_trend=True, add_noise=False) # create climatology
    clim_subset = clim[:,doy_inits:timesteps,:]

    if metric =='deterministic':
        clims = clim.mean(axis=0)
        clim_mean = clims.mean(axis=1)
        clim_std = clims.std(axis=1)
    else:
        clim_subset = np.transpose(np.array([clim_subset[:, i, :].flatten() for i in range(80)]))

    skill_growth_rates = []

    for r in growth_rates:

        skill = []

        for doy in np.arange(0,doy_inits):
            xsim, xobs = generate_data(timesteps=timesteps, growth_rate=r,
                                       sigma=s, phi=0.00, initial_uncertainty=0.00,
                                       doy_0=doy, ensemble_size=15)
            xsim_leads = xsim[:,(doy_inits-doy):(timesteps)]
            xobs_leads = xobs[:,(doy_inits-doy):(timesteps),0]
            if metric =='deterministic':
                xsim_mean = np.mean(xsim_leads, axis=0)
                xsim_abs = abs(xsim_mean - xobs_leads)
                clim_abs = abs(clim_subset - xobs_leads)
                skill.append(np.subtract(xsim_abs, clim_abs))
            else:
                rolling_crps(xobs_leads, xsim_leads)
                rolling_crps(xobs_leads, np.transpose(clim_subset))

        skill = np.array(skill).squeeze() > 0
        skill_growth_rates.append(skill)

    skills_sigma.append(skill_growth_rates)

plt.rcParams.update({'font.size': 16})
fig, (ax1, ax2) = plt.subplots(figsize = (16,9), ncols=2, nrows=2)
pos1 = ax1[0,].imshow(skills_sigma[0][0], cmap='Greys')
ax1[0,].matshow(skills_sigma[0][0], cmap='Greys')
ax1[0,].set_ylabel('Day from Lead')
ax1[0,].set_xlabel('Lead (Day of Year)')
ax1[0,].set_xticks(ticks=np.arange(0,timesteps-doy_inits, step=10), labels=np.arange(doy_inits,timesteps, step=10))
ax1[0,].set_yticks(ticks=np.arange(0,doy_inits, step=5), labels=np.arange(5,doy_inits+5, step=5)[::-1])
plt.colorbar(pos1, ax= ax1[0,], label = 'Skill')
pos2 = ax1[1,].imshow(skills_sigma[0][1], cmap='Greys')
ax1[1,].matshow(skills_sigma[0][1], cmap='Greys')
ax1[1,].set_ylabel('Day from Lead')
ax1[1,].set_xlabel('Lead (Day of Year)')
ax1[1,].set_xticks(ticks=np.arange(0,timesteps-doy_inits, step=10), labels=np.arange(doy_inits,timesteps, step=10))
ax1[1,].set_yticks(ticks=np.arange(0,doy_inits, step=5), labels=np.arange(0,doy_inits, step=5)[::-1])
plt.colorbar(pos2, ax= ax1[1,], label = 'Skill')
pos1 = ax2[0,].imshow(skills_sigma[1][0], cmap='Greys')
ax2[0,].matshow(skills_sigma[1][0], cmap='Greys')
ax2[0,].set_ylabel('Day from Lead')
ax2[0,].set_xlabel('Lead (Day of Year)')
ax2[0,].set_xticks(ticks=np.arange(0,timesteps-doy_inits, step=10), labels=np.arange(doy_inits,timesteps, step=10))
ax2[0,].set_yticks(ticks=np.arange(0,doy_inits, step=5), labels=np.arange(5,doy_inits+5, step=5)[::-1])
plt.colorbar(pos1, ax= ax2[0,], label = 'Skill')
pos2 = ax2[1,].imshow(skills_sigma[1][1], cmap='Greys')
ax2[1,].matshow(skills_sigma[1][1], cmap='Greys')
ax2[1,].set_ylabel('Day from Lead')
ax2[1,].set_xlabel('Lead (Day of Year)')
ax2[1,].set_xticks(ticks=np.arange(0,timesteps-doy_inits, step=10), labels=np.arange(doy_inits,timesteps, step=10))
ax2[1,].set_yticks(ticks=np.arange(0,doy_inits, step=5), labels=np.arange(0,doy_inits, step=5)[::-1])
plt.colorbar(pos2, ax= ax2[1,], label = 'Skill')
fig.tight_layout()
plt.show()
fig.savefig('results/main/skill_deterministic.pdf')

#association_ref = rolling_corrs(xobs, np.expand_dims(clim[:,0], axis=0))
#association_mod = rolling_corrs(xobs, xsim)
#plt.plot(np.transpose(association_mod), color='blue')
#plt.plot(np.transpose(association_ref), color='red')

#===============#
# probabilistic #
#===============#