import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
import fit_torch as ft
import numpy as np
import os

from utils import create_experiment_folder, create_scenario_folder
from visualisations import plot_horizon_maps, plot_fit2, plot_all_dynamics, plot_horizons
from horizons import get_forecast_horizons, forecast_different_leads
from itertools import product

def run_experiment(experiment_path, process, scenario, fit_model, compute_horizons):

    scenario_path = create_scenario_folder(experiment_path, new_folder_name = f'{process}_{scenario}')

    observation_params, initial_params, true_noise, initial_noise = ft.set_parameters(process=process, scenario=scenario,
                                                                                      dir = scenario_path)

    y_train, y_test, sigma_train, sigma_test, x_train, x_test, climatology = ft.create_observations(years=30,
                                                                                                 observation_params=observation_params,
                                                                                                 true_noise=true_noise)

    # get calibrated parameters
    fitted_values = ft.model_fit(fit_model, scenario_path,
                                 y_train, x_train, sigma_train, initial_params, initial_noise,
                                 samples=1, epochs=15, loss_fun='mse', step_length=10)

    # ================================#
    # Forecast with the one scenario #
    # ================================#


    parameter_samples = ft.get_parameter_samples(fitted_values, uncertainty=0.02)
    yinit, forecast, modelfits = ft.forecast_fitted(y_test, x_test, parameter_samples, initial_params, initial_noise,
                                                 initial_uncertainty=0.01)

    plot_fit2(forecast, y_test, dir = scenario_path, clim=climatology)

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

    fhs, metrics_fh = get_forecast_horizons(compute_horizons, y_test, climatology, forecast, dir = scenario_path)

    plot_horizons(fhs, metrics_fh, dir = scenario_path, show_upper=True)

    # =====================================#
    # Forecasting with the fitted at same lead times #
    # ================================================#


    mat_ricker, mat_climatology, mat_ricker_perfect = forecast_different_leads(y_test, x_test, climatology, modelfits,
                                                                               forecast_days = 110, lead_time = 110)

    plot_horizon_maps(mat_ricker, mat_climatology, mat_ricker_perfect, dir = scenario_path)

if __name__ == "__main__":

    new_experiment = False
    seed = True

    if seed:
        np.random.seed(42)

    directory_path = "results"

    if new_experiment:
        experiment_path = create_experiment_folder(directory_path)
    else:
        experiment_path = os.path.join(directory_path, 'version_230907_1221')

    # ========================#
    # Set simulation setting #
    # ========================#

    process = 'stochastic'
    scenario = 'nonchaotic'
    rows = ['deterministic', 'stochastic']
    cols = ['chaotic', 'nonchaotic']
    scenarios = list(product(rows, cols))

    for s in scenarios:
        run_experiment(experiment_path, process=s[0], scenario= s[1], fit_model = True, compute_horizons = True)

    # Forecast all dynamics
    obs, preds, ref = ft.get_forecast_scenario(dir=experiment_path)
    plot_all_dynamics(obs, preds, ref, dir = experiment_path)