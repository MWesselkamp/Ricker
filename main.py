import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
import fit_torch as ft
import pandas as pd
import numpy as np

from data import ForecastData
from torch.utils.data import DataLoader
from visualisations import plot_losses, plot_posterior, plot_horizon_maps, plot_fit2, plot_all_dynamics, plot_horizons
from CRPS import CRPS
from metrics import mse, rolling_corrs

seed = True
if seed:
    np.random.seed(42)
# ========================#
# Set simulation setting #
# ========================#

fit_model = False
forecast_all = False
compute_horizons = True

process = 'stochastic'
scenario = 'chaotic'

observation_params, initial_params, true_noise, initial_noise = ft.set_parameters(process=process, scenario=scenario)
y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = ft.create_observations(years=30,
                                                                                             observation_params=observation_params,
                                                                                             true_noise=true_noise)

# ============#
# Fit Model  #
# ============#

if fit_model:

    if process == 'stochastic':
        fitted_values, losses = ft.fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples=1,
                                           epochs=20, loss_fun='mse', step_length=10)
    else:
        fitted_values, losses = ft.fit_models(y_train, x_train, sigma_train, initial_params, initial_noise, samples=1,
                                           epochs=20, loss_fun='mse', step_length=20)

    fitted_values = pd.DataFrame(fitted_values)

    fitted_values.to_csv(f'results/{scenario}_{process}/fitted_values.csv')
    plot_posterior(fitted_values, saveto=f'results/{scenario}_{process}')
    plot_losses(losses, loss_fun='mse', saveto=f'results/{scenario}_{process}')

# =======================#
# Forecast all dynamics #
# =======================#

if forecast_all:
    obs, preds, ref = ft.get_forecast_scenario()
    plot_all_dynamics(obs, preds, ref, save=True)

# ================================#
# Forecast with the one scenario #
# ================================#

if 'fitted_values' in globals():
    print(f"Deleting globally set parameter values.")
    del(fitted_values)
    print(f"Loading parameters from previous fit.")
    fitted_values = pd.read_csv(f'results/{scenario}_{process}/fitted_values.csv', index_col=False)
    fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)
else:
    print(f"Loading parameters from previous fit.")
    fitted_values = pd.read_csv(f'results/{scenario}_{process}/fitted_values.csv', index_col=False)
    fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)

parameter_samples = ft.get_parameter_samples(fitted_values, uncertainty=0.02)
yinit, forecast, modelfits = ft.forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                             initial_uncertainty=0.01)

plot_fit2(forecast, y_test, scenario=f"{scenario}", process=f"{process}", clim=climatology, save=False)

# sr = [shapiro(ypreds[:,i])[1] for i in range(ypreds.shape[1])]
# plt.plot(sr)

# fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(6, 8), sharex=True)
# for i in range(7):
#    x = ypreds[:,i]
#    mu = x.mean()
#    sigma = x.std()
#    axes[i].hist(x=climatology[:, i].detach().numpy(),bins=20, alpha=0.5, colors='salmon')
#    axes[i].hist(x, bins=20, alpha=0.5)
#    xs = np.linspace(x.min(), x.max(), num=100)
#    axes[i].plot(xs, stats.norm.pdf(xs, mu, sigma))
# axes[i].vlines(x = y_test[i].detach().numpy(), ymin = 0, ymax = 50)

# for i in range(5):
#    print('Ensemble', ps.crps_ensemble(y_test[i].detach().numpy(), ypreds[:,i]))
#    print('Climatology', ps.crps_ensemble(y_test[i].detach().numpy(), climatology.detach().numpy()[:,i]))

# =============================#
# Forecasting with the fitted #
# =============================#

observation = y_test.detach().numpy()[np.newaxis, :]
reference = climatology.detach().numpy()
reference2 = np.tile(reference[:, -1], (reference.shape[1], 1)).transpose()
obs_perfect = np.mean(forecast, axis=0)[np.newaxis, :]
ref_perfect = np.mean(reference, axis=0)[np.newaxis, :]

save = False
if compute_horizons:
    print(f'Computing forecast horizons for {scenario}_{process} setting')
    metrics_fh = ['corr', 'mae', 'fstats', 'crps']  # ['corr', 'mse', 'mae', 'crps']

    fha_ricker = [ft.get_fh(metric, forecast, observation) for metric in metrics_fh]
    fhp_ricker = [ft.get_fh(metric, forecast, obs_perfect) for metric in metrics_fh]
    fha_reference = [ft.get_fh(metric, reference, observation) for metric in metrics_fh]
    fhp_reference = [ft.get_fh(metric, reference, ref_perfect) for metric in metrics_fh]

    metrics_fsh = ['crps']  # ['mse', 'mae', 'crps']
    fsh = [None, None, None] + [ft.get_fsh(forecast, reference, observation, fh_metric=m)[0] for m in metrics_fsh]
    fsh2 = [None, None, None] + [ft.get_fsh(forecast, reference2, observation, fh_metric=m)[0] for m in metrics_fsh]

    fhs = pd.DataFrame([fha_ricker, fhp_ricker, fha_reference, fhp_reference, fsh], columns=metrics_fh,
                       index=['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
    if save:
        fhs.to_csv(f'results/{scenario}_{process}/horizons.csv')
else:
    print(f'LOADING forecast horizons from previous run of {scenario}_{process} setting')
    metrics_fh = ['corr', 'mae', 'fstats', 'crps']
    fhs = pd.read_csv(f'results/{scenario}_{process}/horizons.csv', index_col=0)
    fha_ricker = fhs.loc['fha_ricker'].to_numpy()
    fhp_ricker = fhs.loc['fhp_ricker'].to_numpy()
    fha_reference = fhs.loc['fha_reference'].to_numpy()
    fhp_reference = fhs.loc['fhp_reference'].to_numpy()
    fsh = fhs.loc['fsh'].to_numpy()

plot_fit2(forecast, y_test, scenario=f"{scenario}", process=f"{process}",
          clim=climatology, save=save)

if scenario == 'nonchaotic':
    plot_horizons(fha_ricker, fha_reference, fhp_ricker, fhp_reference, fsh, metrics_fh, scenario, process,
                  save=save, show_upper=True)
else:
    plot_horizons(fha_ricker, fha_reference, fhp_ricker, fhp_reference, fsh, metrics_fh, scenario, process,
                  save=save, show_upper=True)

# =====================================#
# Forecasting with the fitted at same lead times #
# ================================================#

forecast_days = 110
lead_time = 110
data = ForecastData(y_test, x_test, climatology, forecast_days=forecast_days, lead_time=lead_time)
forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

mat_ricker = np.full((lead_time, forecast_days), np.nan)
mat_ricker_perfect = np.full((lead_time, forecast_days), np.nan)
mat_climatology = np.full((lead_time, forecast_days), np.nan)
mat_climatology_perfect = np.full((lead_time, forecast_days), np.nan)

i = 0
fh_metric = 'crps'
for states, temps, clim in forecastloader:

    print('I is: ', i)
    N0 = states[:, 0]
    clim = clim.squeeze().detach().numpy()
    forecast = []
    for modelfit in modelfits:
        forecast.append(modelfit.forecast(N0, temps).detach().numpy())
    forecast = np.array(forecast).squeeze()
    states = states.squeeze().detach().numpy()

    if fh_metric == 'crps':
        performance = [CRPS(forecast[:, i], states[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
        mat_ricker[:, i] = performance
        mat_climatology[:, i] = performance_ref

        performance_perfect = [CRPS(forecast[:, i], forecast[:, i].mean(axis=0)).compute()[0] for i in
                               range(forecast.shape[1])]
        performance_climatology_perfect = [CRPS(clim[:, i], forecast[:, i].mean(axis=0)).compute()[0] for i in
                                           range(forecast.shape[1])]
        mat_ricker_perfect[:, i] = performance_perfect
        mat_climatology_perfect[:, i] = performance_climatology_perfect

    i += 1

plot_horizon_maps(mat_ricker, mat_climatology, mat_ricker_perfect, scenario, process, save=True)

# =====================================================#
# Forecasting with the fitted at different lead times #
# =====================================================#

data = ForecastData(y_test, x_test, climatology)
forecastloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

mat_ricker = np.full((len(y_test), len(y_test)), np.nan)
mat_climatology = np.full((len(y_test), len(y_test)), np.nan)

i = 0
fh_metric = 'crps'
for states, temps, clim in forecastloader:

    print('I is: ', i)
    N0 = states[:, 0]
    clim = clim.squeeze().detach().numpy()
    forecast = []
    for modelfit in modelfits:
        forecast.append(modelfit.forecast(N0, temps).detach().numpy())
    forecast = np.array(forecast).squeeze()
    states = states.squeeze().detach().numpy()

    if fh_metric == 'crps':
        performance = [CRPS(forecast[:, i], states[i]).compute()[0] for i in range(forecast.shape[1])]
        performance_ref = [CRPS(clim[:, i], states[i]).compute()[0] for i in range(clim.shape[1])]
        mat_ricker[i, i:] = performance
        mat_climatology[i, i:] = performance_ref
    elif fh_metric == 'mse':
        performance = mse(states[np.newaxis, :], forecast)
        performance_ref = mse(states[np.newaxis, :], clim)
        mat_ricker[i, i:] = performance
        mat_climatology[i, i:] = performance_ref
    elif fh_metric == 'correlation':
        w = 3
        performance = np.mean(rolling_corrs(states[np.newaxis, :], forecast, window=w), axis=0)
        performance_ref = np.mean(rolling_corrs(states[np.newaxis, :], clim, window=w), axis=0)
        mat_ricker[i, :i + w] = np.nan
        mat_ricker[i, i + w:] = performance
        mat_climatology[i, :i + w] = np.nan
        mat_climatology[i, i + w:] = performance_ref
    i += 1

mat_ricker[np.isinf(mat_ricker)] = np.nan
mat_ricker_plot = mat_ricker
if fh_metric == 'nashsutcliffe':
    fh = mat_ricker <= 0
    mask = np.isfinite(mat_ricker)
    mat_ricker_plot[mask] = fh[mask]
fig, ax = plt.subplots()
plt.imshow(mat_ricker_plot)
plt.colorbar()
plt.xlabel('Day of year')
plt.ylabel('Forecast length')
# plt.savefig(f'plots/horizonmap_ricker_{process}_{scenario}_{fh_metric}fh.pdf')
plt.close()

mat_climatology[np.isinf(mat_climatology)] = np.nan
mat_climatology_plot = mat_climatology
if fh_metric == 'nashsutcliffe':
    fh = mat_climatology <= 0
    mask = np.isfinite(mat_climatology)
    mat_climatology_plot[mask] = fh[mask]
fig, ax = plt.subplots()
plt.imshow(mat_climatology_plot)
plt.colorbar()
plt.xlabel('Day of year')
plt.ylabel('Forecast length')
# plt.savefig(f'plots/horizonmap_climatology_{process}_{scenario}_{fh_metric}fh.pdf')
plt.close()

if fh_metric != 'correlation':
    fig, ax = plt.subplots()
    skill = 1 - (mat_ricker / mat_climatology)  # If mse of climatology is larger, term is larger zero.
    fh = skill <= 0  # FH of the ricker is reached when mse of climatology drops below ricker mse
    mask = np.isfinite(skill)
    skill[mask] = fh[mask]
    plt.imshow(skill, cmap='autumn_r')
    plt.colorbar()
    plt.xlabel('Day of year')
    plt.ylabel('Forecast length')
    # plt.savefig(f'plots/horizonmap_skill_{process}_{scenario}_{fh_metric}fh.pdf')
else:
    fig, ax = plt.subplots()
    skill = mat_ricker  # If mse of climatology is larger, term is larger zero.
    fh = skill < 0.5  # FH of the ricker is reached when mse of climatology drops below ricker mse
    mask = np.isfinite(skill)
    skill[mask] = fh[mask]
    plt.imshow(skill, cmap='autumn_r')
    plt.colorbar()
    plt.xlabel('Day of year')
    plt.ylabel('Forecast length')

plt.plot([np.argmax(skill[i, i:]) for i in range(skill.shape[0])])
