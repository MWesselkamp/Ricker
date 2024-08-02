import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
import numpy as np
import os
import random
import scipy

from scipy.stats import t, f
from metrics import anomaly, variance_standard_error, t_statistic_two_samples
from metrics import rolling_corrs, tstat_inverse, correlation_standard_error
from utils import create_scenario_folder
from properscoring import crps_gaussian

random.seed(42)
np.random.seed(42)

def correlation_based_horizon(obs, preds, spin_up = 3, dir=''):

        # Compute rolling correlations
        c = rolling_corrs(obs.reshape(1,len(obs)), preds, window=10)

        c_mean = np.mean(c, axis=0)
        c_mean_se = np.array([correlation_standard_error(c_mean[i], 10) for i in range(c_mean.shape[0])]).T
        c_mean_se_upper = c_mean + 2 * c_mean_se
        c_mean_se_lower = c_mean - 2 * c_mean_se

        # Determine critical value to use as threshold
        critical_ts = scipy.stats.t.ppf(1 - 0.05/ 2, 10)
        threshold = np.round(tstat_inverse(critical_ts, samples=10), 4)

        plt.figure(figsize=(7, 5))
        plt.plot(c_mean.T, linestyle='-', color='black')
        plt.plot(c_mean_se_upper.T, linestyle='-', linewidth=0.6, color='green')
        plt.plot(c_mean_se_lower.T, linestyle='-', linewidth=0.6, color='green')
        plt.hlines(y=threshold, xmin=0, xmax=len(c_mean), colors='blue', linestyles='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'correlation.pdf'))
        plt.close()

        def get_horizon(value, threshold, spin_up):
            if np.any(value < threshold):
                if np.argmax(value < threshold) <= spin_up:
                    return np.argmax(value[spin_up:] < threshold)
                else:
                    return np.argmax(value < threshold)
            else:
                return len(value)

        # Get forecast horizon
        h_mean = get_horizon(c_mean, threshold, spin_up)
        h_upper = get_horizon(c_mean_se_upper, threshold, spin_up)
        h_lower = get_horizon(c_mean_se_lower, threshold, spin_up)
        print('Correlation based forecast horizon with uncertainties:', h_mean, h_upper, h_lower)
        horizons = [h_mean, h_upper, h_lower]

        return {'metrics': {'mean':c_mean, 'mean_sd':c_mean_se, 'upper':c_mean_se_upper, 'lower':c_mean_se_lower},
                'horizons': h_mean,
                'thresholds': threshold}

def anomaly_quantile_horizon(obs, preds, dir = ''):

        an = anomaly(obs, preds)
        an_mean = np.mean(an, axis=0)
        an_var = np.var(an, axis=0)
        an_var_sd = (2 * np.sqrt(an_var) ** 4) / (an.shape[0] - 1)
        an_mean_upper = an_mean + 2 * np.sqrt(an_var) + 2 * an_var_sd
        an_mean_lower = an_mean - 2 * np.sqrt(an_var) - 2 * an_var_sd

        # Determine critical value to use as threshold: confidence interval of 95% total anomaly
        upper_threshold = an.mean() + 2 * an.std()
        lower_threshold = an.mean() - 2 * an.std()

        plt.figure(figsize=(7, 5))
        plt.plot(an_mean, linestyle='-', color='black')
        plt.plot(an_mean_upper, linestyle='-', linewidth=0.6, color='green')
        plt.plot(an_mean_lower, linestyle='-', linewidth=0.6, color='green')
        plt.hlines(y=upper_threshold, xmin=0, xmax=len(an_mean), colors='blue', linestyles='--')
        plt.hlines(y=lower_threshold, xmin=0, xmax=len(an_mean), colors='blue', linestyles='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'anomaly_quantile.pdf'))
        plt.close()

        # Get forecast horizon
        def get_horizon(value, upper_threshold = None, lower_threshold = None):
            if (np.any((value > upper_threshold) | (value < lower_threshold))):
                return np.argmax(((value > upper_threshold)| (value < lower_threshold)))
            else:
                return len(value)

        h_mean = get_horizon(an_mean, upper_threshold, lower_threshold)
        h_upper = get_horizon(an_mean_upper, upper_threshold, lower_threshold)
        h_lower = get_horizon(an_mean_lower, upper_threshold, lower_threshold)
        h_upper = max(h_mean, h_upper)
        h_lower = min(h_mean, h_lower)
        print('Anomaly based forecast horizon with uncertainties:', h_mean, h_upper, h_lower)
        horizons = [h_mean, h_upper, h_lower]
        threshold = abs(upper_threshold)

        return {'metrics': {'mean':an_mean, 'mean_sd':an_var, 'upper':an_mean_upper, 'lower':an_mean_lower},
                'horizons': h_mean,
                'thresholds': threshold}
def anomaly_quantile_skill_horizon(obs, preds, ref):

        out_ricker = anomaly_quantile_horizon(obs, preds)
        out_reference = anomaly_quantile_horizon(obs, ref)

        mean_skill = out_ricker[('anomaly')][0] / out_reference['anomaly'][0]

        mean_skill_sigmaAB = np.corrcoef(out_ricker['anomaly'][0], out_reference['anomaly'][0])[0, 1] * out_ricker['anomaly'][1] * \
                             out_reference['anomaly'][1]
        skill_sd = mean_skill ** 2 * ((out_ricker['anomaly'][1] / out_ricker['anomaly'][0]) ** 2 + (
                    out_reference['anomaly'][1] / out_reference['anomaly'][0]) ** 2) - 2 * mean_skill_sigmaAB / (
                               out_ricker['anomaly'][0] * out_reference['anomaly'][0])

        upper_skill = mean_skill + 2 * skill_sd
        lower_skill = mean_skill - 2 * skill_sd

        plt.figure(figsize=(7, 5))
        plt.plot(mean_skill, linestyle='-', color='black')
        plt.plot(upper_skill, linestyle='-', linewidth=0.6, color='green')
        plt.plot(lower_skill, linestyle='-', linewidth=0.6, color='green')
        plt.close()

        def get_horizon(value, threshold=1):
            if np.any(value < threshold):
                if np.argmax(value < threshold) <= 1:
                    return np.argmax(value[1:] < threshold)
                else:
                    return np.argmax(value < threshold)
            else:
                return len(value)

        hmean = get_horizon(mean_skill, 1)
        hupper = get_horizon(upper_skill, 1)
        hlower = get_horizon(lower_skill, 1)

        print('Anomaly based forecast skill horizon with uncertainties:', hmean)

        return {'horizons': [hmean, hupper, hlower],
                'thresholds': 1}

def cumulative_fstatistics_skill_horizon(obs, preds, ref, spin_up = 5, dir = ''):

        an_preds = anomaly(obs, preds)
        an_ref = anomaly(obs, ref)

        # Compute within time step variance of anomalies
        # we skip the first step here because we cannot compute the variance of a single value with the total variance
        var_within = [np.var(an_preds[:, i]) for i in range(an_preds.shape[1])]
        # Compute standard error of within time step variance
        var_within_sd = np.array([variance_standard_error(var_within[i], an_preds.shape[0]) for i in range(len(var_within))])

        # Compute total variance of anomalies
        var_reference = np.array([np.var(an_ref[:, i]) for i in range(an_ref.shape[1])])
        var_reference_sd = np.array(
            [variance_standard_error(var_reference[i], an_ref.shape[0]) for i in range(len(var_reference))])

        var_within_samples = np.array(
            [np.random.normal(var_within[i], var_within_sd[i], 1000) for i in range(len(var_within))])
        var_total_samples = np.array(
            [np.random.normal(var_reference[i], var_reference_sd[i], 1000) for i in range(len(var_reference))])

        rhoAB = np.array([np.corrcoef(var_within_samples[i], var_total_samples[i])[0, 1] for i in range(len(var_within))])
        sigmaAB = np.array([var_within_sd[i] * var_reference_sd[i] * rhoAB[i] for i in range(len(var_within))])

        # Compute F-statistics, the function to which we add the error.
        # Null hypothesis: var_reference = var_within
        # Alternative hypothesis:  var_within > var_reference
        fstats_mean = np.array([(var_within[i] / var_reference[i]) for i in range(len(var_within))])

        # Compute variance of F-statistics based on https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        fstats_var = np.array([(fstats_mean[i] ** 2 *
                                ((var_within_sd[i] / var_within[i]) ** 2 +
                                 (var_reference_sd[i] / var_reference[i]) ** 2 -
                                 2 * sigmaAB[i] / (var_reference[i] * var_within[i]))) for i in range(len(var_within))])
        fstats_sd = np.sqrt(fstats_var)

        # Compute upper and lower bounds of F-statistics
        fstats_upper = fstats_mean + 2 * fstats_sd
        fstats_lower = fstats_mean - 2 * fstats_sd

        # Calculate the threshold based on the critical F-statistic value
        thresholds = np.array([f.ppf(1 - 0.05 / 2, (an_preds.shape[0] - 1), (len(var_reference[:i])+1)*an_ref.shape[0]) for i in range(var_reference.shape[0])])
        # pvals = [f.cdf(stats[i], df1, df2) for i in range(len(stats))]
        # reject Null if F is larger than critical value

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 7))
        ax1.plot(an_preds.T[:, (spin_up - 1):], linestyle='-', color='purple', alpha=0.6)
        ax1.plot(an_ref.T[:, (spin_up - 1):], linestyle='-', color='magenta', alpha=0.1)
        ax1.set_ylim(-2, 2)
        ax1.set_ylabel('Anomaly')
        ax2.plot(var_within[(spin_up - 1):], linestyle='-', linewidth=1.4, color='purple', label='forecast')
        ax2.plot(var_reference[(spin_up - 1):], linestyle='-', linewidth=1.4, color='magenta', label='reference')
        ax2.set_ylabel('Variances')
        ax2.legend(loc='upper right')
        ax3.plot(fstats_mean[(spin_up - 1):], linestyle='-', color='black')
        ax3.plot(fstats_upper[(spin_up - 1):], linestyle='-', linewidth=1.1, color='green')
        ax3.plot(fstats_lower[(spin_up - 1):], linestyle='-', linewidth=1.1, color='green')
        ax3.plot(thresholds[(spin_up - 1):], linestyle='--', linewidth=1.3, color='blue', label='threshold')
        ax3.legend(loc='upper right')
        ax3.set_ylabel('F-statistic')
        ax3.set_xlabel('Forecast time')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'fstats_skill.pdf'))
        plt.close()

def cumulative_fstatistics_horizon(obs, preds, spin_up = 1, interval = 1, dir = ''):

        an = anomaly(obs, preds)

        # Compute within time step variance of anomalies
        # we skip the first step here because we cannot compute the variance of a single value with the total variance
        var_within = [np.var(an[:, i]) for i in range(1, an.shape[1])]
        # Compute standard error of within time step variance
        var_within_sd = np.array([variance_standard_error(var_within[i], an.shape[0]) for i in range(len(var_within))])

        # Compute total variance of anomalies
        var_total = np.array([np.var(an[:, :i]) for i in range(1, an.shape[1])])
        var_total_sd = np.array(
            [variance_standard_error(var_total[i], (len(var_total[:i])+1)*an.shape[0]) for i in range(len(var_total))])

        var_within_samples = np.array(
            [np.random.normal(var_within[i], var_within_sd[i], 1000) for i in range(len(var_within))])
        var_total_samples = np.array(
            [np.random.normal(var_total[i], var_total_sd[i], 1000) for i in range(len(var_total))])

        rhoAB = np.array([np.corrcoef(var_within_samples[i], var_total_samples[i])[0, 1] for i in range(len(var_within))])
        sigmaAB = np.array([var_within_sd[i] * var_total_sd[i] * rhoAB[i] for i in range(len(var_within))])

        # Compute F-statistics, the function to which we add the error.
        fstats_mean = np.array([(var_within[i] / var_total[i]) for i in range(len(var_within))])

        # Compute variance of F-statistics based on https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        fstats_var = np.array([(fstats_mean[i] ** 2 *
                                ((var_within_sd[i] / var_within[i]) ** 2 +
                                 (var_total_sd[i] / var_total[i]) ** 2 -
                                 2 * sigmaAB[i] / (var_total[i] * var_within[i]))) for i in range(len(var_within))])
        fstats_sd = np.sqrt(fstats_var)

        # Compute upper and lower bounds of F-statistics
        fstats_upper = fstats_mean + 2 * fstats_sd
        fstats_lower = fstats_mean - 2 * fstats_sd

        # Calculate the threshold based on the critical F-statistic value
        thresholds = np.array([f.ppf(1 - 0.05 / 2, (an.shape[0] - 1), (len(var_total[:i])+1)*an.shape[0]) for i in range(var_total.shape[0])])
        # pvals = [f.cdf(stats[i], df1, df2) for i in range(len(stats))]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 7))
        ax1.plot(an.T[:,(spin_up - 1):], linestyle='-', color='gray')
        ax1.set_ylabel('Anomaly')
        ax2.plot(var_within[(spin_up-1):], linestyle='-', linewidth=1.2, color='purple', label='forecast')
        ax2.plot(var_total[(spin_up - 1):], linestyle='-', linewidth=1.2, color='magenta', label='cumulative')
        ax2.set_ylabel('Variances')
        ax2.legend(loc='upper right')
        ax3.plot(fstats_mean[(spin_up - 1):], linestyle='-', color='black')
        ax3.plot(fstats_upper[(spin_up - 1):], linestyle='-', linewidth=1.1, color='green')
        ax3.plot(fstats_lower[(spin_up - 1):], linestyle='-', linewidth=1.1, color='green')
        ax3.plot(thresholds[(spin_up - 1):], linestyle='-', linewidth=1.1, color='blue', label='threshold')
        ax3.legend(loc='upper right')
        ax3.set_ylabel('F-statistic')
        ax3.set_xlabel('Forecast time')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'fstats_cumulative.pdf'))
        plt.close()

        # Get forecast horizon
        def get_horizon(value, threshold, spin_up, interval = 1):
            if np.any(value > threshold):
                if interval == 1:
                    if np.argmax(value > threshold) <= spin_up:
                        return np.argmax(value[spin_up:] > threshold[spin_up:])
                    else:
                        return np.argmax(value > threshold)
                else:
                    threshold_reached = value < threshold
                    # Define a kernel for convolution (moving window of size 3)
                    kernel = np.ones(interval)
                    # Use convolution to sum up elements in the moving window
                    result = np.convolve(threshold_reached.astype(int), kernel, mode='valid')
                    return np.argmax(result == 3)
            else:
                return len(value)

        h_mean = get_horizon(fstats_mean, thresholds, spin_up)
        h_upper = get_horizon(fstats_upper, thresholds, spin_up)
        h_lower = get_horizon(fstats_lower, thresholds, spin_up)

        print('Cumulative fstatistics forecast horizon with uncertainties:', h_mean, h_upper, h_lower)
        horizons = [h_mean, h_upper, h_lower]

        return {'horizons': h_mean,
                'metrics': {'mean':fstats_mean,'mean_sd':fstats_sd, 'upper':fstats_upper, 'lower':fstats_lower},
                'thresholds': thresholds,
                'additionals': {'anomaly': an,'var_within':var_within,'var_total':var_total} }


def cumulative_tstatistics_skill_horizon(obs, preds, ref, spin_up=10, dir=''):
    """
    Calculate the cumulative T-statistics skill horizon.

    Parameters:
    obs (np.ndarray): Observed data.
    preds (np.ndarray): Predicted data.
    ref (np.ndarray): Reference data.
    spin_up (int): Spin-up period.
    dir (str): Directory to save the plot.

    Returns:
    dict: A dictionary containing horizons, mean T-statistics, and thresholds.
    """

    def get_horizon(t_stat, threshold):
        """
        Determine the forecast horizon based on the T-statistics and threshold.

        Parameters:
        t_stat (np.ndarray): Array of T-statistics.
        threshold (float): T-statistic threshold value.

        Returns:
        int: Forecast horizon.
        """
        if np.any(np.abs(t_stat) > threshold):
            return np.argmax(np.abs(t_stat) > threshold)
        return len(t_stat)

    # Calculate anomalies
    an_preds = anomaly(obs, preds)
    an_ref = anomaly(obs, ref)

    # Compute T-statistics for each time step
    t_stat = np.array([
        t_statistic_two_samples(an_preds[:, (i - spin_up):i], an_ref[:, (i - spin_up):i], omega_0=0.0)
        for i in range(spin_up, an_preds.shape[1])
    ])

    # Degrees of freedom
    df = an_preds.shape[0] * spin_up + an_ref.shape[0] * spin_up - 2

    # Calculate critical T-value
    threshold = t.ppf(1 - 0.5 * 0.05, df)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

    # Plot anomalies
    ax1.plot(an_preds.T[:, (spin_up - 1):], linestyle='-', color='purple', alpha=0.6, label='Predicted Anomalies')
    ax1.plot(an_ref.T[:, (spin_up - 1):], linestyle='-', color='magenta', alpha=0.3, label='Reference Anomalies')
    ax1.set_ylabel('Anomaly')
    ax1.legend(loc='upper right')

    # Plot T-statistics
    ax2.plot(t_stat, linestyle='-', color='black')
    ax2.axhline(threshold, xmin=0, xmax=len(t_stat), linestyle='-', linewidth=1.1, color='blue', label='Threshold')
    ax2.set_ylabel('T-statistic')
    ax2.set_xlabel('Forecast Time')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'tstats.pdf'))
    plt.close()

    # Calculate forecast horizon
    h_mean = get_horizon(t_stat, threshold)

    print('T-statistics forecast horizon:', h_mean)

    return {
        'horizons': h_mean,
        'metrics': {'mean': t_stat},
        'thresholds': threshold
    }


def fstatistics_horizon(obs, preds, dir=''):

        an = anomaly(obs, preds)

        # Compute within time step variance of anomalies
        var_within = [np.var(an[:, i]) for i in range(an.shape[1])]
        # Compute standard error of within time step variance
        var_within_sd = np.array([variance_standard_error(var_within[i], an.shape[0]) for i in range(len(var_within))])

        # Compute total variance of anomalies
        var_total = np.var(an)
        # Compute standard error of total variance
        var_total_sd = variance_standard_error(var_total, len(var_within))

        # Compute covariance between within time step variance and total variance
        # Sample from normal distribution with mean and standard deviation of total variance for correlation
        var_total_samples = np.random.normal(var_total, var_total_sd, 1000)
        # Sample from normal distribution with mean and standard deviation of within time step variance for correlation
        var_with_samples = np.array([np.random.normal(var_within[i], var_within_sd[i], 1000) for i in range(len(var_within))])
        rhoAB = np.array([np.corrcoef(var_with_samples[i], var_total_samples)[0,1] for i in range(len(var_within))])
        sigmaAB = np.array([var_within_sd[i]*var_total_sd*rhoAB[i] for i in range(len(var_within))])

        # Compute F-statistics, the function to which we add the error.
        fstats_mean = np.array([(var_within[i] / var_total) for i in range(len(var_within))])

        # Compute variance of F-statistics based on https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        fstats_var = np.array([(fstats_mean[i]**2 *
                                ((var_within_sd[i] / var_within[i])**2  +
                                (var_total_sd / var_total)**2 -
                                2* sigmaAB[i] / (var_total * var_within[i]))) for i in range(len(var_within))])
        fstats_sd = np.sqrt(fstats_var)

        # Compute upper and lower bounds of F-statistics
        fstats_upper = fstats_mean + 2 * fstats_sd
        fstats_lower = fstats_mean - 2 * fstats_sd

        # Calculate the threshold based on the critical F-statistic value
        df1, df2 = 1000 - 1, an.shape[1] - 1
        threshold = f.ppf(1 - 0.05 / 2, df1, df2)
        # pvals = [f.cdf(stats[i], df1, df2) for i in range(len(stats))]

        plt.figure(figsize=(7, 5))
        plt.plot(fstats_mean, linestyle='-', color='black')
        plt.plot(fstats_upper, linestyle='-', linewidth=0.6, color='green')
        plt.plot(fstats_lower, linestyle='-', linewidth=0.6, color='green')
        plt.hlines(y=threshold, xmin=0, xmax=len(fstats_mean), colors='blue', linestyles='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'fstats.pdf'))
        plt.close()

        # Get forecast horizon
        def get_horizon(value, threshold):
            if np.any(value > threshold):
                return np.argmax(value > threshold)
            else:
                return len(value)

        h_mean = get_horizon(fstats_mean, threshold)
        h_upper = get_horizon(fstats_upper, threshold)
        h_lower = get_horizon(fstats_lower, threshold)

        print('Forecast horizon with uncertainties:', h_mean, h_upper, h_lower)
        horizons = [h_mean, h_upper, h_lower]

        return {'horizons': h_mean,
                'metrics': {'mean':fstats_mean, 'mean_sd':fstats_sd, 'upper':fstats_upper, 'lower':fstats_lower},
                'thresholds':threshold}

def crps_horizon(obs, preds, dir = ''):


        # Fit the Gaussian distribution to the data
        mu_init = np.array([np.mean(preds[:, i]) for i in range(len(preds[0, :]))])
        mu_init_sd = np.array([np.std(preds[:, i])/np.sqrt(len(preds[:, 0])) for i in range(len(preds[0, :]))])

        sigma_init = np.array([np.sqrt(np.var(preds[:, i])) for i in range(len(preds[0, :]))])
        sigma_init_sd = np.array([(sigma_init[i]*0.5*variance_standard_error(np.var(preds[:, i]), len(preds[:, 0]))/np.var(preds[:, i]))**2 for i in range(len(preds[0, :]))])

        # Sample mu and sigma from normal distribution
        mu_samples = np.array([np.random.normal(mu_init[i], mu_init_sd[i], 500) for i in range(len(mu_init))])
        sigma_samples = np.array([np.random.normal(sigma_init[i], sigma_init_sd[i], 500) for i in range(len(sigma_init))])

        # Compute CRPS for each sample
        crps_samples = np.array([crps_gaussian(obs[i], mu_samples[i, :], sigma_samples[i, :]) for i in range(len(mu_init))])
        crps_mean = crps_samples.mean(axis=1)
        crps_std = crps_samples.std(axis=1)
        crps_upper = crps_mean + 2 * crps_std
        crps_lower = crps_mean - 2 * crps_std
        # crpsg = [crps_gaussian(np.random.normal(obs_example[i], obs_example[i]*0.1, 1000), mu_samples[i, :].mean(), sigma_samples[i, :].mean()) for i in range(mu_samples.shape[0])]
        # crpsg = np.array(crpsg)
        # plt.hist(crpsg)
        # threshold = np.array([np.quantile(crpsg[i,:], 0.05) for i in range(crpsg.shape[0])])

        threshold = 0.025

        plt.figure(figsize=(7, 5))
        plt.plot(crps_mean, linestyle='-', color='black')
        plt.plot(crps_upper, linestyle='-', linewidth=0.6, color='green')
        plt.plot(crps_lower, linestyle='-', linewidth=0.6, color='green')
        plt.hlines(y=threshold, xmin=0, xmax=len(crps_mean), colors='blue', linestyles='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'crps.pdf'))
        plt.close()

        def get_horizon(value, threshold):
            if np.any(value > threshold):
                return np.argmax(value > threshold)
            else:
                return len(value)

        h_mean = get_horizon(crps_mean, threshold)
        h_upper = get_horizon(crps_upper, threshold)
        h_lower = get_horizon(crps_lower, threshold)

        print('Forecast horizon with uncertainties:', h_mean, h_upper, h_lower)
        horizons = [h_mean, h_upper, h_lower]

        return  {'horizons': h_mean,
                 'metrics':{'mean':crps_mean, 'mean_sd':crps_std, 'upper':crps_upper, 'lower':crps_lower},
                 'thresholds': threshold}

def crps_skill_horizon(obs, preds, ref, interval, dir = ''):

        out_ricker = crps_horizon(obs, preds, dir)
        out_reference = crps_horizon(obs, ref, dir)

        mean_skill = out_ricker['metrics']['mean'] / out_reference['metrics']['mean']

        mean_skill_sigmaAB = np.corrcoef(out_ricker['metrics']['mean'], out_reference['metrics']['mean'])[0, 1] * out_ricker['metrics']['mean_sd'] * out_reference['metrics']['mean_sd']
        skill_sd = mean_skill**2*((out_ricker['metrics']['mean_sd']/out_ricker['metrics']['mean'])**2 + (out_reference['metrics']['mean_sd']/out_reference['metrics']['mean'])**2) - 2*mean_skill_sigmaAB/(out_ricker['metrics']['mean']*out_reference['metrics']['mean'])

        upper_skill = mean_skill + 2*skill_sd
        lower_skill = mean_skill - 2*skill_sd

        threshold = 1

        plt.figure(figsize=(7, 5))
        plt.plot(mean_skill, linestyle='-', color='black')
        plt.plot(upper_skill, linestyle='-', linewidth=0.6, color='green')
        plt.plot(lower_skill, linestyle='-', linewidth=0.6, color='green')
        plt.hlines(y=threshold, xmin=0, xmax=len(mean_skill), colors='blue', linestyles='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'crps_skill.pdf'))
        plt.close()
        def get_horizon(value, threshold, interval):
            if np.any(value < threshold):
                if interval == 1:
                    if np.argmax(value < threshold) <= 1:
                        return np.argmax(value[1:] < threshold)
                    else:
                        return np.argmax(value < threshold)
                else:
                    threshold_reached = value < threshold
                    # Define a kernel for convolution (moving window of size 3)
                    kernel = np.ones(interval)
                    # Use convolution to sum up elements in the moving window
                    result = np.convolve(threshold_reached.astype(int), kernel, mode='valid')
                    return np.argmax(result == 3)
            else:
                return len(value)

        hmean = get_horizon(mean_skill, threshold, interval=1)
        hupper = get_horizon(upper_skill, threshold, interval=1)
        hlower = get_horizon(lower_skill, threshold, interval=1)

        hmean_interval = get_horizon(mean_skill, threshold, interval=interval)
        hupper_interval = get_horizon(upper_skill, threshold, interval=interval)
        hlower_interval = get_horizon(lower_skill, threshold, interval=interval)

        print('Forecast skill horizon with uncertainties:', hmean, hupper, hlower)
        horizons = [hmean, hupper, hlower]
        interval_horizons = [hmean_interval, hupper_interval, hlower_interval]

        return {'horizons': hmean,
                'metrics': {'mean':mean_skill, 'mean_sd':skill_sd, 'upper':upper_skill, 'lower':lower_skill},
                'thresholds': threshold,
                'interval_horizons':hmean_interval}

class Experiment():

    def __init__(self, obs, preds, ref, dir = ''):

        self.obs = obs
        self.preds = preds
        self.ref = ref
        self.dir = dir

        create_scenario_folder(dir, new_folder_name='actual_ricker')
        create_scenario_folder(dir, new_folder_name='actual_reference')
        create_scenario_folder(dir, new_folder_name='intrinsic_ricker')
        create_scenario_folder(dir, new_folder_name='intrinsic_reference')

    def compute_horizons(self):
        def call_horizon_functions(y, yhat, type, dir=''):

            h_correlation = correlation_based_horizon(y, yhat, spin_up=3, dir=os.path.join(dir, type))
            h_anomaly = anomaly_quantile_horizon(y, yhat, dir=os.path.join(dir, type))
            h_fstat = cumulative_fstatistics_horizon(y, yhat, spin_up=2, dir=os.path.join(dir, type))
            h_crps = crps_horizon(y, yhat, dir=os.path.join(dir, type))

            return {'correlation': h_correlation,
                    'anomaly': h_anomaly,
                    'fstat': h_fstat,
                    'crps': h_crps}

        self.actual_ricker = call_horizon_functions(self.obs, self.preds, type = 'actual_ricker', dir = self.dir)
        self.actual_reference = call_horizon_functions(self.obs, self.ref, type = 'actual_reference', dir = self.dir)
        self.intrinsic_ricker = call_horizon_functions(self.preds.mean(axis=0), self.preds,
                                                       type = 'intrinsic_ricker', dir = self.dir)
        self.intrinsic_reference = call_horizon_functions(self.ref.mean(axis=0), self.ref,
                                                          type = 'intrinsic_reference', dir = self.dir)
        self.skill_ricker = crps_skill_horizon(self.obs, self.preds, self.ref, interval=5, dir = self.dir)

    def assemble_horizons(self, interval = 1):
        def extract_horizons(h_correlation, h_anomaly, h_fstat, h_crps):
            horizons = [h_correlation['horizons'],
                        h_anomaly['horizons'],
                        h_fstat['horizons'],
                        h_crps['horizons']]

            return horizons

        if interval == 1:
            skill_horizon = self.skill_ricker['horizons']
        else:
            skill_horizon = self.skill_ricker['interval_horizons']

        self.horizons = [extract_horizons(self.actual_ricker['correlation'],
                                               self.actual_ricker['anomaly'],
                                               self.actual_ricker['fstat'],
                                               self.actual_ricker['crps']),
                         extract_horizons(self.intrinsic_ricker['correlation'],
                                               self.intrinsic_ricker['anomaly'],
                                               self.intrinsic_ricker['fstat'],
                                               self.intrinsic_ricker['crps']),
                         extract_horizons(self.actual_reference['correlation'],
                                               self.actual_reference['anomaly'],
                                               self.actual_reference['fstat'],
                                               self.actual_reference['crps']),
                         extract_horizons(self.intrinsic_reference['correlation'],
                                               self.intrinsic_reference['anomaly'],
                                               self.intrinsic_reference['fstat'],
                                               self.intrinsic_reference['crps']),
                         [None, None, None, skill_horizon]]

    def assemble_thresholds(self, full = True):

        def extract_thresholds(h_correlation, h_anomaly, h_fstat, h_crps):
            thresholds = [h_correlation['thresholds'],
                        h_anomaly['thresholds'],
                        (h_fstat['thresholds'] if full else np.mean(h_fstat['thresholds'])),
                        h_crps['thresholds']]

            return thresholds

        self.thresholds = {'actual_ricker':extract_thresholds(self.actual_ricker['correlation'],
                                               self.actual_ricker['anomaly'],
                                               self.actual_ricker['fstat'],
                                               self.actual_ricker['crps']),
                         'intrinsic_ricker':extract_thresholds(self.intrinsic_ricker['correlation'],
                                               self.intrinsic_ricker['anomaly'],
                                               self.intrinsic_ricker['fstat'],
                                               self.intrinsic_ricker['crps']),
                         'actual_reference':extract_thresholds(self.actual_reference['correlation'],
                                               self.actual_reference['anomaly'],
                                               self.actual_reference['fstat'],
                                               self.actual_reference['crps']),
                         'intrinsic_reference':extract_thresholds(self.intrinsic_reference['correlation'],
                                               self.intrinsic_reference['anomaly'],
                                               self.intrinsic_reference['fstat'],
                                               self.intrinsic_reference['crps']),
                         'skill':[None, None, None, self.skill_ricker['thresholds']]}

    def assemble_proficiency(self):

        def extract_metrics(h_correlation, h_anomaly, h_fstat, h_crps):
            metrics = [h_correlation['metrics'],
                        h_anomaly['metrics'],
                        h_fstat['metrics'],
                        h_crps['metrics']]

            return metrics

        self.proficiencies = {'actual_ricker':extract_metrics(self.actual_ricker['correlation'],
                                               self.actual_ricker['anomaly'],
                                               self.actual_ricker['fstat'],
                                               self.actual_ricker['crps']),
                         'intrinsic_ricker':extract_metrics(self.intrinsic_ricker['correlation'],
                                               self.intrinsic_ricker['anomaly'],
                                               self.intrinsic_ricker['fstat'],
                                               self.intrinsic_ricker['crps']),
                         'actual_reference':extract_metrics(self.actual_reference['correlation'],
                                               self.actual_reference['anomaly'],
                                               self.actual_reference['fstat'],
                                               self.actual_reference['crps']),
                         'intrinsic_reference':extract_metrics(self.intrinsic_reference['correlation'],
                                               self.intrinsic_reference['anomaly'],
                                               self.intrinsic_reference['fstat'],
                                               self.intrinsic_reference['crps']),
                         'skill':[None, None, None, self.skill_ricker['metrics']]}

    def plot_assembled_proficiencies(self, type, color_mean = '#8B7355',color_sd = '#CDAA7D'):

        proficiencies = self.proficiencies[type]
        correlation_mean = proficiencies[0]['mean']
        correlation_upper = proficiencies[0]['upper']
        correlation_lower = proficiencies[0]['lower']
        anomaly_mean = proficiencies[1]['mean']
        anomaly_upper = proficiencies[1]['upper']
        anomaly_lower = proficiencies[1]['lower']
        fstat_mean = proficiencies[2]['mean']
        fstat_upper = proficiencies[2]['upper']
        fstat_lower = proficiencies[2]['lower']
        crps_mean = proficiencies[3]['mean']
        crps_upper = proficiencies[3]['upper']
        crps_lower = proficiencies[3]['lower']

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [4,1,1,1,1]})
        ref = ax1.plot(np.transpose(self.ref), color='gray', label = 'Long-term\n Mean', zorder=0)
        obs = ax1.plot(np.transpose(self.obs), color='magenta', label='Observed', zorder=1)
        fit = ax1.fill_between(np.arange(self.preds.shape[1]), self.preds.transpose().min(axis=1),
                              self.preds.transpose().max(axis=1), color='b', alpha=0.4)
        fit_mean = ax1.plot(self.preds.transpose().mean(axis=1), color='b', alpha=0.5, label='Ricker')
        plt.setp(ref[1:], label="_")
        #plt.setp(fit_mean[1:], label="_")
        plt.setp(obs[1:], label="_")
        ax1.legend()
        ax1.set_ylabel('Relative size', weight='bold')
        ax1.yaxis.set_label_coords(-0.1, 0.5)
        ax2.plot(anomaly_mean, linestyle='-', color=color_mean)
        ax2.plot(anomaly_upper, linestyle='-', linewidth=0.9, color=color_sd)
        ax2.plot(anomaly_lower, linestyle='-', linewidth=0.9, color=color_sd)
        ax2.hlines(y=self.thresholds[type][1], xmin=0, xmax=len(correlation_mean), color='black',linewidth=0.8, linestyle='--')
        ax2.hlines(y=-self.thresholds[type][1], xmin=0, xmax=len(correlation_mean), color='black',linewidth=0.8, linestyle='--')
        ax2.set_ylabel('Anom', weight='bold')
        ax2.yaxis.set_label_coords(-0.1, 0.5)
        ax3.plot(correlation_mean, linestyle='-', color=color_mean)
        ax3.plot(correlation_upper, linestyle='-', linewidth=1, color=color_sd)
        ax3.plot(correlation_lower, linestyle='-', linewidth=1, color=color_sd)
        ax3.hlines(y=self.thresholds[type][0], xmin=0, xmax=len(correlation_mean), color='black',linewidth=0.8, linestyle='--')
        ax3.set_ylabel('Corr',  weight='bold')
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        ax4.plot(fstat_mean[1:], linestyle='-', color=color_mean)
        ax4.plot(fstat_upper[1:], linestyle='-', linewidth=1, color=color_sd)
        ax4.plot(fstat_lower[1:], linestyle='-', linewidth=1, color=color_sd)
        ax4.plot(self.thresholds[type][2][1:], color='black',linewidth=0.8, linestyle='--')
        ax4.set_ylabel('F-Stat',  weight='bold')
        ax4.yaxis.set_label_coords(-0.1, 0.5)
        ax5.plot(crps_mean[1:], linestyle='-', color=color_mean)
        ax5.plot(crps_upper[1:], linestyle='-', linewidth=1, color=color_sd)
        ax5.plot(crps_lower[1:], linestyle='-', linewidth=1, color=color_sd)
        ax5.hlines(y=self.thresholds[type][3], xmin=0, xmax=len(correlation_mean), color='black',linewidth=0.8, linestyle='--')
        ax5.set_ylabel('CRPS', weight='bold')
        ax5.set_xlabel('Time [Generation]', weight='bold')
        ax5.yaxis.set_label_coords(-0.1, 0.5)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.13)  # You can adjust the value as needed
        plt.show()
        plt.savefig(os.path.join(os.path.join(self.dir, type), 'assembled_proficiencies.pdf'))
        plt.close()

    def make_plots(self):

        self.assemble_proficiency()
        self.assemble_thresholds()

        self.plot_assembled_proficiencies(type = 'actual_ricker')
        self.plot_assembled_proficiencies(type = 'intrinsic_ricker')
        self.plot_assembled_proficiencies(type = 'actual_reference')
        self.plot_assembled_proficiencies(type = 'intrinsic_reference')



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

