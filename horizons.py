import numpy as np
import scipy
import pandas as pd
import os

from data import ForecastData
from torch.utils.data import DataLoader
from CRPS import CRPS
from scipy.stats import f
from metrics import mse, rolling_corrs, rmse, absolute_differences, fstat, tstat_inverse
from sklearn.metrics import r2_score

def pointwise_evaluation(forecast, observation, fh_metric, **kwargs):
    """
    Evaluate forecast performance based on a specified metric.

    Parameters:
    - forecast (numpy.ndarray): The forecasted values.
    - observation (numpy.ndarray): The observed values.
    - fh_metric (str): The forecast evaluation metric to use.
    - **kwargs: Additional keyword arguments, if needed for certain metrics.

    Returns:
    - numpy.ndarray: An array containing performance scores based on the specified metric.
    """

    # Check the selected forecast evaluation metric
    if fh_metric == 'crps':
        try:
            # Calculate CRPS (Continuous Ranked Probability Score) for each point in the forecast
            performance = [CRPS(forecast[:, i], observation[i]).compute()[0] for i in range(forecast.shape[1])]
        except ValueError:
            # Handle the case where observation may need to be squeezed
            performance = [CRPS(forecast[:, i], observation.squeeze()[i]).compute()[0] for i in range(forecast.shape[1])]

    elif fh_metric == 'ae':
        # Calculate the absolute error between forecast and observation
        performance = np.subtract(observation, forecast)

    elif fh_metric == 'mae':
        # Calculate the mean absolute error
        performance = np.mean(absolute_differences(observation, forecast), axis=0)

    elif fh_metric == 'mse':
        # Calculate the mean squared error
        performance = mse(observation, forecast)

    elif fh_metric == 'rmse':
        # Calculate the root mean squared error
        performance = rmse(observation, forecast)

    elif fh_metric == 'rsquared':
        # Calculate R-squared values for various forecast lengths
        performance = [[r2_score(observation[:j], forecast[i, :j]) for i in range(forecast.shape[0])] for j in range(1, forecast.shape[1])]

    elif fh_metric == 'corr':
        # Calculate rolling correlations with a specified window size (w)
        w = kwargs['w']
        performance = np.mean(rolling_corrs(observation, forecast, window=w), axis=0)

    elif fh_metric == 'fstats':
        # Calculate F-statistics and p-values for the forecast and observation
        fstats, pvals = fstat(forecast, observation)
        performance = fstats

    # Return the computed performance scores as a numpy array
    return np.array(performance)

def set_threshold(fh_metric, **kwargs):
    """
    Set a threshold value based on the specified forecast evaluation metric.

    Parameters:
    - fh_metric (str): The forecast evaluation metric to use.
    - **kwargs: Additional keyword arguments, which may vary depending on the metric.

    Returns:
    - float: The computed threshold value.
    """

    # Check the selected forecast evaluation metric
    if fh_metric == 'mae':
        # Calculate the threshold based on the standard deviation of anomalies/residuals
        anomaly = (kwargs['forecast'] - kwargs['observation'])
        threshold = anomaly.std()
        print('Standard deviation of Anomaly/Residuals:', threshold)

    elif fh_metric == 'fstats':
        # Calculate the threshold based on the critical F-statistic value
        df1, df2 = kwargs['forecast'].shape[0] - 1, kwargs['forecast'].shape[1] - 1
        threshold = f.ppf(1 - kwargs['alpha'] / 2, df1, df2)
        print(f"Critical F at alpha = {kwargs['alpha']}:", threshold)

    elif fh_metric == 'corr':
        # Calculate the threshold based on the critical t-statistic value for correlations
        critical_ts = scipy.stats.t.ppf(1 - kwargs['alpha'] / 2, kwargs['w'])
        threshold = np.round(tstat_inverse(critical_ts, samples=kwargs['w']), 4)
        print(f"Critical r at alpha = {kwargs['alpha']}:", threshold)

    elif fh_metric == "crps":
        # Set a fixed threshold value for CRPS
        threshold = 0.05

    elif fh_metric == 'mse':
        # Set a fixed threshold value for Mean Squared Error (MSE)
        threshold = 0.025

    elif fh_metric == 'rmse':
        # Set a fixed threshold value for Root Mean Squared Error (RMSE)
        threshold = 0.025

    return threshold

def forecast_skill_horizon(performance, performance_ref, fh_metric, tolerance=0.05):
    """
    Determine the forecast skill horizon (FSH) based on performance metrics.

    Parameters:
    - performance (numpy.ndarray): Performance scores for the current forecast.
    - performance_ref (numpy.ndarray): Reference performance scores for a benchmark or ideal forecast.
    - fh_metric (str): The forecast evaluation metric used.
    - tolerance (float): The tolerance level to determine when FSH is reached.

    Returns:
    - int: The FSH (Forecast Skill Horizon) indicating when performance surpasses the tolerance level.
    - numpy.ndarray: Skill scores representing the difference between reference and current performance.
    """

    # Calculate the skill as the difference between reference and current performance
    skill = performance_ref - performance

    # Check the selected forecast evaluation metric to determine FSH
    if fh_metric in ["crps", "mae", "mse", "rmse", "fstats"]:
        reached_fsh = skill < -tolerance
    else:
        raise ValueError(f"Unsupported forecast evaluation metric: {fh_metric}. Supported metrics are {', '.join(['crps', 'mae', 'mse', 'rmse', 'fstats'])}.")

    # Find the FSH by identifying the index where the skill surpasses the tolerance level
    if reached_fsh.any():
        fsh = np.argmax(reached_fsh)
    else:
        fsh = len(reached_fsh)

    return fsh, skill

def forecast_horizon(performance, fh_metric, threshold):
    """
        Determine the forecast horizon (FH) based on performance metrics and a threshold.

        Parameters:
        - performance (numpy.ndarray): Performance scores for the current forecast.
        - fh_metric (str): The forecast evaluation metric used.
        - threshold (float): The threshold value to determine when FH is reached.

        Returns:
        - int: The FH (Forecast Horizon) indicating when performance surpasses the threshold.
    """

    # Check if the metric is "corr" (correlation), where a higher value is better
    if fh_metric != "corr":
        reached_fh = performance > threshold
    else:
        # only correlation is better if larger
        reached_fh = performance < threshold

    # Find the FH by identifying the index where the performance surpasses the threshold
    if reached_fh.any():
        fh = np.argmax(reached_fh)
    else:
        fh = len(reached_fh)

    return fh

def get_fh(fh_metric, forecast, observation):
    """
    Compute the forecast horizon (FH) for a given forecast evaluation metric, forecast, and observation.

    Parameters:
    - fh_metric (str): The forecast evaluation metric to use.
    - forecast (numpy.ndarray): The forecasted values.
    - observation (numpy.ndarray): The observed values.

    Returns:
    - int: The FH (Forecast Horizon) indicating when the specified metric surpasses a threshold.
    """

    # Calculate performance scores for the specified metric
    performance = pointwise_evaluation(forecast, observation, fh_metric=fh_metric, w=5)

    # Set a threshold for the specified metric
    threshold = set_threshold(fh_metric, forecast=forecast, observation=observation, w=5, alpha=0.05)

    # Determine the forecast horizon based on the performance and threshold
    fh = forecast_horizon(performance, fh_metric, threshold)

    return fh


def get_fsh(forecast, reference, obs, fh_metric):
    """
    Calculate the Forecast Skill Horizon (FSH) based on forecast performance and reference performance.

    Parameters:
    - forecast (numpy.ndarray): The forecasted values to be evaluated.
    - reference (numpy.ndarray): The reference forecasted values (e.g., ideal or benchmark forecasts).
    - obs (numpy.ndarray): The observed values.
    - fh_metric (str): The forecast evaluation metric to use.

    Returns:
    - int: The FSH (Forecast Skill Horizon) indicating when the forecast skill surpasses a tolerance level.
    - numpy.ndarray: Skill scores representing the difference between reference and forecast performance.
    """

    # Calculate performance scores for the forecast and reference forecast
    performance_forecast = pointwise_evaluation(forecast, obs, fh_metric=fh_metric)
    performance_reference = pointwise_evaluation(reference, obs, fh_metric=fh_metric)

    # Calculate the FSH and skill scores based on the two sets of performance scores
    fsh, skill = forecast_skill_horizon(performance_forecast, performance_reference, fh_metric=fh_metric)

    return fsh, skill

def get_forecast_horizons(compute_horizons, y_test, climatology, forecast, dir):
    """
    Calculate and save forecast horizons based on different performance metrics, or load them if already computed.

    Parameters:
    - compute_horizons (bool): Whether to compute forecast horizons or load them.
    - y_test (torch.Tensor): Observed values.
    - climatology (torch.Tensor): Climatology reference values.
    - forecast (numpy.ndarray): The forecasted values.
    - dir (str): The directory for saving/loading forecast horizons.

    Returns:
    - pd.DataFrame: DataFrame containing forecast horizons for various metrics.
    - list: List of metrics used for calculating forecast horizons.
    """

    # Convert tensors to numpy arrays
    observation = y_test.detach().numpy()[np.newaxis, :]
    reference = climatology.detach().numpy()
    reference2 = np.tile(reference[:, -1], (reference.shape[1], 1)).transpose()
    obs_perfect = np.mean(forecast, axis=0)[np.newaxis, :]
    ref_perfect = np.mean(reference, axis=0)[np.newaxis, :]

    if compute_horizons:
        print(f'Computing forecast horizons and saving in {dir}')
        metrics_fh = ['corr', 'mae', 'fstats', 'crps']

        # Calculate forecast horizons for various metrics
        fha_ricker = [get_fh(metric, forecast, observation) for metric in metrics_fh]
        fhp_ricker = [get_fh(metric, forecast, obs_perfect) for metric in metrics_fh]
        fha_reference = [get_fh(metric, reference, observation) for metric in metrics_fh]
        fhp_reference = [get_fh(metric, reference, ref_perfect) for metric in metrics_fh]

        metrics_fsh = ['crps']
        fsh = [None, None, None] + [get_fsh(forecast, reference, observation, fh_metric=m)[0] for m in metrics_fsh]
        fsh2 = [None, None, None] + [get_fsh(forecast, reference2, observation, fh_metric=m)[0] for m in metrics_fsh]

        # Create a DataFrame to store forecast horizons
        fhs = pd.DataFrame([fha_ricker, fhp_ricker, fha_reference, fhp_reference, fsh], columns=metrics_fh,
                           index=['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
        # Save the DataFrame to a CSV file
        fhs.to_csv(os.path.join(dir, 'horizons.csv'))

    else:
        print(f'LOADING forecast horizons from {dir}')
        metrics_fh = ['corr', 'mae', 'fstats', 'crps']
        # Load the DataFrame with forecast horizons from a CSV file
        fhs = pd.read_csv(os.path.join(dir, 'horizons.csv'), index_col=0)

    return fhs, metrics_fh


def forecast_different_leads(y_test, x_test, climatology, modelfits, forecast_days = 110, lead_time = 110):

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

    return mat_ricker, mat_climatology, mat_ricker_perfect


def mean_forecastskill(forecast_skill, threshold):

    mean_skill = np.mean(forecast_skill, axis=0)
    mean_fhs = np.array([i > threshold for i in mean_skill])
    return mean_fhs, None, None

def forecastskill_mean(forecast_skill, threshold):

    fhs = np.array([i > threshold for i in forecast_skill])
    fhs_mean = np.mean(fhs.astype(int), axis=0)  # for plotting
    fhs_var = np.std(fhs.astype(int), axis=0)
    return fhs_mean, fhs_var, fhs

# Quantile Horizon
def efh_quantile(metric, accepted_error, actual_error, timesteps, quantiles = (0.01, 0.99), ps = False):
    """
    1. Function parameter: What quantiles to use?
    2. What is the "expected error"
    """
    # Petcheys empirical Confidence Intervalls.
    def empCL(x, percent):
        ex = np.sort(x)[np.floor(percent / 100 * len(x)).astype(int)]
        return (ex)
    q_lower = [empCL(actual_error[:, i], quantiles[0]*100) for i in range(actual_error.shape[1])]
    q_mid = [empCL(actual_error[:, i], 50) for i in range(actual_error.shape[1])]
    q_upper = [empCL(actual_error[:, i], quantiles[1]*100) for i in range(actual_error.shape[1])]
    # Simply taking Quantiles
    error_metrics = ['mse', 'abs_diff']
    qu = np.quantile(actual_error, (quantiles[0], quantiles[1]), axis=0)
    efh = []
    for i in range(timesteps):
        if metric in error_metrics:
            e = not (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        elif metric == 'cor':
            e = (min(qu[0, i], qu[1, i]) < accepted_error < max(qu[0, i], qu[1, i])) | ((min(qu[0, i], qu[1, i]) < accepted_error) & (max(qu[0, i], qu[1, i]) < accepted_error))
        efh.append(e)
    if np.sum(efh) == 0:
        min_pred_skill = timesteps
    else:
        min_pred_skill = min(np.arange(len(efh))[efh])
    if ps:
        return min_pred_skill
    else:
        return efh, min_pred_skill

