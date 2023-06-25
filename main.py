import numpy as np
import matplotlib.pyplot as plt

from simulations import generate_data
from references import diurnal_climatology
from metrics import rolling_corrs, rolling_crps

np.random.seed(42)

def sample_ensemble_member(xsim):

    index = np.random.randint(0, xsim.shape[0], 1)
    control = xsim[index, :]
    ensemble = np.delete(xsim, index, axis=0)

    return control, ensemble, index

def metric_map_intrinsic(growth_rates, timesteps, doy_inits, metric, bootstrap_samples):

    if metric == 'deterministic':
        s = 0.00
    elif metric == 'stochastic':
        s = 0.001

    metric_map = []
    for r in growth_rates:
        qualities = []
        for doy in np.arange(0, doy_inits):
            bs_qualities = []
            for bs in range(bootstrap_samples):
                xsim, xobs = generate_data(timesteps=timesteps, growth_rate=r,
                                           sigma=s, phi=0.00, initial_uncertainty=0.00,
                                           doy_0=doy, ensemble_size=15)
                # replace observation by sample from forecast ensemble
                xobs, xsim, index = sample_ensemble_member(xsim)

                xsim_leads = xsim[:, (doy_inits - doy):(timesteps)]
                xobs_leads = xobs[:, (doy_inits - doy):(timesteps)]

                if metric == 'deterministic':
                    #quality = np.mean(rolling_corrs(xobs_leads, xsim_leads), axis=0)
                    quality = np.mean(abs(np.subtract(xobs_leads, xsim_leads)), axis=0)
                elif metric == 'stochastic':
                    quality = rolling_crps(xobs_leads, xsim_leads)
                bs_qualities.append(quality)
            # take the bootstrap mean
            bs_qualities = np.mean(np.array(bs_qualities), axis=0)
            qualities.append(bs_qualities)
        qualities_matrix = np.array(qualities)

        metric_map.append(qualities_matrix)

    return metric_map


def metric_map(growth_rates, timesteps, doy_inits, metric, my_dir = ''):

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
                #quality = np.mean(rolling_corrs(xobs_leads, xsim_leads), axis=0)
                quality = np.mean(abs(np.subtract(xobs_leads, xsim_leads)), axis=0)
            elif metric == 'stochastic':
                quality = rolling_crps(xobs_leads, xsim_leads)
            qualities.append(quality)
        qualities_matrix = np.array(qualities)

        metric_map.append(qualities_matrix)

    return metric_map

def get_climatology(scenario, timesteps, doy_inits, metric, growth_rate, s, add_trend=True):

    clim = diurnal_climatology(scenario = scenario, growth_rate=growth_rate, sigma=s, initial_uncertainty=0.00, add_trend=add_trend,
                               add_noise=False)  # create climatology
    clim_subset = clim[doy_inits:timesteps, :]

    if metric == 'deterministic':

        clims = clim_subset.mean(axis=1)

    else:
        clims = np.transpose(clim_subset)

    return clims


def plot_simulations(scenario, growth_rate, s, add_trend, add_noise=False, my_dir=''):
    clim = diurnal_climatology(scenario = scenario, growth_rate=growth_rate, sigma=s, initial_uncertainty=0.00, add_trend=True,
                               add_noise=False)  # create climatology
    xsim, xobs = generate_data(timesteps=365, growth_rate=growth_rate,
                               sigma=s, phi=0.00, initial_uncertainty=0.00,
                               doy_0=0, ensemble_size=15,
                               environment='exogeneous', add_trend=add_trend, add_noise=add_noise)  # create simulations
    xobs = xobs[:, :, 0]  # observations

    clim_mean = clim.mean(axis=1)
    clim_std = clim.std(axis=1)

    plt.plot(np.transpose(clim_mean), color='gray', label='climatology')
    plt.plot(clim_mean + 2 * clim_std, color='gray', linestyle='--')
    plt.plot(clim_mean - 2 * clim_std, color='gray', linestyle='--')
    plt.plot(np.transpose(xsim), color='blue')
    plt.plot(np.transpose(xobs), color='red', label='observations')
    plt.legend()
    plt.xlabel("Day of Year")
    plt.ylabel("Relative stock biomass")
    plt.savefig(my_dir)
    plt.tight_layout()
    plt.close()

def metric_map_benchmark(growth_rates, timesteps, doy_inits, metric, threshold = False):

    if metric =='deterministic':
        s = 0.00
    else:
        s = 0.001

    metric_map_benchmark = []

    for r in growth_rates:

        scenario = 'imperfect_model'
        clim_subset = get_climatology(scenario, timesteps, doy_inits, metric, growth_rate=r, s=s, add_trend=False)
        skill = []
        plot_simulations(scenario, r, s, add_trend=True, my_dir = f'results/main/simulations_{s}_{r}.pdf')

        for doy in np.arange(0,doy_inits):
            xsim, xobs = generate_data(timesteps=timesteps, growth_rate=r,
                                       sigma=s, phi=0.00, initial_uncertainty=0.00,
                                       doy_0=doy, ensemble_size=15)
            xsim_leads = xsim[:,(doy_inits-doy):(timesteps)]
            xobs_leads = xobs[:,(doy_inits-doy):(timesteps),0]

            if metric =='deterministic':
                xsim_abs = np.mean(abs(xsim_leads - xobs_leads),axis=0)
                clim_abs = abs(clim_subset - xobs_leads)
                skill.append(np.subtract(xsim_abs, clim_abs))
            else:
                xsim_crps = rolling_crps(xobs_leads, xsim_leads)
                clim_crps = rolling_crps(xobs_leads, clim_subset)
                skill.append(np.subtract(xsim_crps, clim_crps))

        if threshold:
            skill = np.array(skill).squeeze() > 0
        else:
            skill = np.array(skill).squeeze()

        metric_map_benchmark.append(skill)

    return metric_map_benchmark

def metric_map_benchmark_intrinsic(growth_rates, timesteps, doy_inits, metric, bootstrap_samples, threshold = False):

    if metric =='deterministic':
        s = 0.00
    else:
        s = 0.001

    metric_map_benchmark_intrinsic = []

    for r in growth_rates:

        scenario = 'perfect_model'
        clim_subset = get_climatology(scenario, timesteps, doy_inits, metric, r, s, add_trend=False)

        skill = []
        plot_simulations(scenario, r, s, add_trend=True, my_dir = f'results/main/simulations_int_{s}_{r}.pdf')

        for doy in np.arange(0,doy_inits):
            bs_skill = []
            for bs in range(bootstrap_samples):
                xsim, xobs = generate_data(timesteps=timesteps, growth_rate=r,
                                           sigma=s, phi=0.00, initial_uncertainty=0.00,
                                           doy_0=doy, ensemble_size=15)
                # replace observation by sample from forecast ensemble
                xobs, xsim, index = sample_ensemble_member(xsim)

                xsim_leads = xsim[:,(doy_inits-doy):(timesteps)]
                xobs_leads = xobs[:,(doy_inits-doy):(timesteps)]

                if metric =='deterministic':
                    xsim_abs = np.mean(abs(xsim_leads - xobs_leads),axis=0)
                    clim_abs = abs(clim_subset - xobs_leads)
                    bs_skill.append(np.subtract(xsim_abs, clim_abs))
                else:
                    xsim_crps = rolling_crps(xobs_leads, xsim_leads)
                    clim_crps = rolling_crps(xobs_leads, clim_subset)
                    bs_skill.append(np.subtract(xsim_crps, clim_crps))
            bs_skill = np.mean(np.array(bs_skill), axis=0)
            skill.append(bs_skill)

        if threshold:
            skill = np.array(skill).squeeze() > 0
        else:
            skill = np.array(skill).squeeze()

        metric_map_benchmark_intrinsic.append(skill)

    return metric_map_benchmark_intrinsic


def plot_metric_maps(deterministic_map, stochastic_map, labels=['Absolute error', 'CRPS'], colorbars = True, my_dir=''):
    plt.rcParams.update({'font.size': 16})
    fig, (ax1, ax2) = plt.subplots(figsize=(16, 9), ncols=2, nrows=2)
    pos1 = ax1[0,].imshow(deterministic_map[0], cmap='Greys')
    ax1[0,].matshow(deterministic_map[0], cmap='Greys')
    ax1[0,].set_ylabel('Day from Lead')
    ax1[0,].set_xlabel('Lead (Day of Year)')
    ax1[0,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                       labels=np.arange(doy_inits, timesteps, step=10))
    ax1[0,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits + 5, step=5)[::-1])
    if colorbars:
        plt.colorbar(pos1, ax=ax1[0,], label=labels[0])
    pos2 = ax1[1,].imshow(deterministic_map[1], cmap='Greys')
    ax1[1,].matshow(deterministic_map[1], cmap='Greys')
    ax1[1,].set_ylabel('Day from Lead')
    ax1[1,].set_xlabel('Lead (Day of Year)')
    ax1[1,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                       labels=np.arange(doy_inits, timesteps, step=10))
    ax1[1,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits+5, step=5)[::-1])
    if colorbars:
        plt.colorbar(pos2, ax=ax1[1,], label=labels[0])
    pos1 = ax2[0,].imshow(stochastic_map[0], cmap='Greys')
    ax2[0,].matshow(stochastic_map[0], cmap='Greys')
    ax2[0,].set_ylabel('Day from Lead')
    ax2[0,].set_xlabel('Lead (Day of Year)')
    ax2[0,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                       labels=np.arange(doy_inits, timesteps, step=10))
    ax2[0,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits + 5, step=5)[::-1])
    if colorbars:
        plt.colorbar(pos1, ax=ax2[0,], label=labels[1])
    pos2 = ax2[1,].imshow(stochastic_map[1], cmap='Greys')
    ax2[1,].matshow(stochastic_map[1], cmap='Greys')
    ax2[1,].set_ylabel('Day from Lead')
    ax2[1,].set_xlabel('Lead (Day of Year)')
    ax2[1,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                       labels=np.arange(doy_inits, timesteps, step=10))
    ax2[1,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits+5, step=5)[::-1])
    if colorbars:
        plt.colorbar(pos2, ax=ax2[1,], label=labels[1])
    fig.tight_layout()
    plt.show()
    fig.savefig(my_dir)
    plt.close()

def plot_metric_map_intersection(similarity_nonchaotic, similarity_chaotic, my_dir):
    plt.rcParams.update({'font.size': 16})
    fig, (ax1) = plt.subplots(figsize=(16, 5), ncols=2, nrows=1)
    ax1[0,].imshow(similarity_nonchaotic, cmap='winter_r')
    ax1[0,].set_ylabel('Day from Lead')
    ax1[0,].set_xlabel('Lead (Day of Year)')
    ax1[0,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                           labels=np.arange(doy_inits, timesteps, step=10))
    ax1[0,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits + 5, step=5)[::-1])
    ax1[1,].imshow(similarity_chaotic, cmap='winter_r')
    ax1[1,].set_ylabel('Day from Lead')
    ax1[1,].set_xlabel('Lead (Day of Year)')
    ax1[1,].set_xticks(ticks=np.arange(0, timesteps - doy_inits, step=10),
                           labels=np.arange(doy_inits, timesteps, step=10))
    ax1[1,].set_yticks(ticks=np.arange(0, doy_inits, step=5), labels=np.arange(5, doy_inits+5, step=5)[::-1])
    fig.tight_layout()
    plt.show()
    fig.savefig(my_dir)
    plt.close()
#============================#
# Practical Forecast horizon #
#=============================#

growth_rates = [0.1, 0.95]
timesteps = 120
doy_inits = 40

absolute_error = metric_map(growth_rates, timesteps, doy_inits, metric='deterministic')
crps = metric_map(growth_rates, timesteps, doy_inits, metric='stochastic')

plot_metric_maps(absolute_error, crps, labels = ['Absolute error', 'CRPS'], my_dir = 'results/main/accuracy.pdf')

#========================#
# Forecast skill horizon #
#========================#

# benchmark against climatology.

absolute_error_skill = metric_map_benchmark(growth_rates, timesteps, doy_inits, metric='deterministic', threshold=True)
crps_skill = metric_map_benchmark(growth_rates, timesteps, doy_inits, metric='stochastic', threshold=True)

plot_metric_maps(absolute_error_skill, crps_skill, labels = ['Absolute error skill', 'CRPS skill'], colorbars=False, my_dir = 'results/main/skill.pdf')


#============================#
# Intrinsic Forecast horizon #
#============================#

absolute_error_intrinsic = metric_map_intrinsic(growth_rates, timesteps, doy_inits, metric='deterministic', bootstrap_samples=20)
crps_intrinsic = metric_map_intrinsic(growth_rates, timesteps, doy_inits, metric='stochastic', bootstrap_samples=20)

plot_metric_maps(absolute_error_intrinsic, crps_intrinsic, labels = ['Absolute error', 'CRPS'], my_dir = 'results/main/accuracy_intrinsic.pdf')

#==================================#
# Intrinsic forecast skill horizon #
#==================================#

# benchmark against climatology.

absolute_error_skill_intrinsic = metric_map_benchmark_intrinsic(growth_rates, timesteps, doy_inits, metric='deterministic', bootstrap_samples=20, threshold=True)
crps_skill_intrinsic  = metric_map_benchmark_intrinsic(growth_rates, timesteps, doy_inits, metric='stochastic', bootstrap_samples=20,threshold=True)

plot_metric_maps(absolute_error_skill_intrinsic, crps_skill_intrinsic, labels = ['Absolute error skill', 'CRPS skill'], colorbars=False, my_dir = 'results/main/skill_intrinsic.pdf')


#=======================================#
# Intrinsic vs. practical skill horizon #
#=======================================#

similarity_nonchaotic = np.logical_and(crps[0], crps_skill_intrinsic[0])
similarity_chaotic = np.logical_and(crps[1], crps_skill_intrinsic[1])

plot_metric_map_intersection(similarity_nonchaotic, similarity_chaotic, my_dir = 'results/main/skill_intersection.pdf')

prob_nonchaotic = np.sum(similarity_nonchaotic)/similarity_nonchaotic.size
prob_chaotic = np.sum(similarity_chaotic)/similarity_chaotic.size