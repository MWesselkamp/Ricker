import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# The script is focused on visualizing critical correlation values for different scenarios,
# considering varying significance levels, sample sizes, and confidence intervals.
# It provides insights into the relationship between correlation, sample size, and statistical significance.

# Set global font size for plots
plt.rcParams['font.size'] = 18

def tstat(correlation, samples):
    """
    Calculate t-statistic for correlation coefficient.

    Parameters:
    - correlation (float): Correlation coefficient.
    - samples (int): Number of samples.

    Returns:
    - float: T-statistic.
    """
    return correlation * np.sqrt((samples - 2) / (1 - correlation ** 2))

def tstat_inverse(t, samples):
    """
    Calculate inverse of t-statistic.

    Parameters:
    - t (float): T-statistic.
    - samples (int): Number of samples.

    Returns:
    - float: Inverse of t-statistic.
    """
    return t / np.sqrt((samples - 2 + t ** 2))


def plot_correlation_threshold(sample_size, critical_rs, critical_rs2, critical_rs3):
    """
    Plot correlation threshold for different significance levels and sample sizes.

    Parameters:
    - sample_size (numpy.ndarray): Array of sample sizes.
    - critical_rs (numpy.ndarray): Array of critical correlation values for alpha = 0.05.
    - critical_rs2 (numpy.ndarray): Array of critical correlation values for alpha = 0.2.
    - critical_rs3 (numpy.ndarray): Array of critical correlation values for alpha = 0.35.
    """
    sr = np.argmax(critical_rs < 0.5)

    plt.figure(figsize=(7, 6))
    plt.plot(sample_size, critical_rs, color='blue', label=f"alpha = 0.05")
    plt.plot(sample_size, critical_rs2, color='orange', label=f"alpha = 0.2")
    plt.plot(sample_size, critical_rs3, color='salmon', label=f"alpha = 0.35")

    # Vertical lines and annotations
    plt.vlines(3, 0.1, critical_rs[3], colors="black", linestyles="--")
    plt.hlines(critical_rs[3], 0, 3, colors="black", linestyles="--")
    plt.vlines(10, 0.1, critical_rs[10], colors="gray", linestyles="--")
    plt.hlines(critical_rs[10], 0, 10, colors="gray", linestyles="--")
    plt.vlines(sr, 0.1, critical_rs[sr], colors="lightgray", linestyles="--")
    plt.hlines(critical_rs[sr], 0, sr, colors="lightgray", linestyles="--")

    # Text annotations
    plt.text(x=4, y=critical_rs[3], s=f'{critical_rs[3]}')
    plt.text(x=11, y=critical_rs[10], s=f'{critical_rs[10]}')
    plt.text(x=sr + 1, y=critical_rs[sr], s=f'{critical_rs[sr]}')

    # Axis labels and legend
    plt.xlabel("Length of moving window\n (Sample size)")
    plt.ylabel("Critical $r$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/correlation_threshold.pdf')
    plt.close()


def plot_correlation_threshold_confidence(significance_level, critical_rs, critical_rs2):
    """
    Plot correlation threshold confidence for different significance levels and sample sizes.

    Parameters:
    - significance_level (numpy.ndarray): Array of significance levels.
    - critical_rs (numpy.ndarray): Array of critical correlation values for sample size 10.
    - critical_rs2 (numpy.ndarray): Array of critical correlation values for sample size 3.
    """
    sr = np.argmax(critical_rs < 0.5)
    sr2 = np.argmax(critical_rs2 < 0.5)

    plt.figure(figsize=(7, 6))
    plt.plot(significance_level, critical_rs, color='blue', label=f"sample size = 10")
    plt.plot(significance_level, critical_rs2, color='orange', label=f"sample size = 3")

    # Vertical lines and annotations
    plt.vlines(significance_level[sr], 0, 0.5, colors="black", linestyles="--")
    plt.vlines(significance_level[sr2], 0, 0.5, colors="black", linestyles="--")
    plt.hlines(critical_rs[sr], 0, significance_level[sr], colors="black", linestyles="--")
    plt.hlines(critical_rs2[sr2], 0, significance_level[sr2], colors="black", linestyles="--")

    # Text annotations
    plt.text(x=significance_level[sr], y=0.55, s=f'{significance_level[sr]}')
    plt.text(x=significance_level[sr2], y=0.55, s=f'{significance_level[sr2]}')

    # Axis labels and legend
    plt.xlabel("Significance Level alpha")
    plt.ylabel("Critical $r$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/correlation_threshold_confidence.pdf')
    plt.close()


# Set the window size
w = 10

# Generate correlation values and corresponding t-statistics
corr = np.linspace(0, 1, 100)
tstats = [tstat(corr[i], w) for i in range(len(corr))]

# Generate an array of sample sizes
sample_size = np.arange(0, 50)

# Calculate critical correlation values for different significance levels
critical_rs = np.round(tstat_inverse(scipy.stats.t.ppf(1 - 0.05 / 2, sample_size), samples=sample_size), 4)
critical_rs2 = np.round(tstat_inverse(scipy.stats.t.ppf(1 - 0.2 / 2, sample_size), samples=sample_size), 4)
critical_rs3 = np.round(tstat_inverse(scipy.stats.t.ppf(1 - 0.35 / 2, sample_size), samples=sample_size), 4)

# Plot correlation threshold for different significance levels and sample sizes
plot_correlation_threshold(sample_size, critical_rs, critical_rs2, critical_rs3)

# Generate an array of significance levels
significance_level = np.round(np.linspace(0.0, 1.0, 200), 2)

# Calculate critical correlation values for significance levels and sample sizes
critical_ts = scipy.stats.t.ppf(1 - significance_level / 2, w)
critical_rs = np.round(tstat_inverse(critical_ts, samples=w), 4)
critical_rs2 = np.round(tstat_inverse(scipy.stats.t.ppf(1 - significance_level / 2, 3), samples=3), 4)

# Plot correlation threshold confidence for different significance levels and sample sizes
plot_correlation_threshold_confidence(significance_level, critical_rs, critical_rs2)
