import numpy as np

def efh_mean(metric, profiencies, threshold, ps = False):
    """
    1. Function parameter: threshold.
    """
    profiencies_mean = profiencies.mean(axis=0)

    if metric == 'corr':
        efh = np.array([i < threshold for i in profiencies])
        pred_skills = np.argmax(profiencies < threshold, axis=1)
        #mean_pred_skill = min(np.arange(profiencies.shape[1])[profiencies_mean < threshold])
    elif metric in ['mse', 'abs_diff']:
        efh = np.array([i > threshold for i in profiencies])
        pred_skills = np.argmax(profiencies > threshold, axis=1)
        # pred_skills = [min(np.arange(profiencies.shape[1])[efh[i,:]]) for i in range(profiencies.shape[0])]
        # mean_pred_skill = min(np.arange(profiencies.shape[1])[profiencies_mean > threshold])
    b = [np.sum(efh, axis=1) == 0]  # get the rows where the efh is never reached
    pred_skills[b] = profiencies.shape[1] # replace by maximum efh

    if ps:
        return pred_skills
    else:
        return efh, pred_skills#, mean_pred_skill

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

