import numpy as np
import models
import matplotlib.pyplot as plt

uncertainties = {"parameters":False, "initial":True,"observation":False,"stoch":True}
# Set hyperparameters.
hp_r = {"iterations":30, "initial_size": 0.8, "initial_uncertainty": 1e-1, "ensemble_size": 25}
# Set parameters
theta_r2 = {'lambda':np.exp(1.8), 'alpha':1, 'sigma':0.1} # lambda = exp(r)

# Initialize model 2
ricker = models.Ricker_Single(uncertainties)
ricker.set_parameters(theta_r2)
simu = ricker.simulate(hp_r)
x = simu["ts"]
ricker.visualise(np.transpose(x))

# compute the square-error-based SNR (Czanner et al. 2015, PNAS)
# this is an estimator for the true expected SNR.
def ss_based_SNR(obs, pred):
    """
    The squared error based SNR, an estimator of the true expected SNR.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Independent of sample size!
    """
    EPE_mean = np.dot(np.transpose(obs - np.mean(obs)), (obs - np.mean(obs)))
    EPE_model = np.dot(np.transpose(obs - pred), (obs - pred))
    signal = (EPE_mean - EPE_model)
    noise = EPE_model

    return signal/noise

# we create the perfect model setting by treating one trajectory from the simulated ensemble as observation.
# we bootstrap this procedure to propagate uncertainty in initial conditions and thereby get a distribution of the SNR.
bs_samples = 50
bs_arr_modelworld = np.zeros((hp_r['iterations'], bs_samples))
for j in range(bs_samples):
    obs_ind, pred_ind = np.random.randint(hp_r['ensemble_size'], size=2)
    print(obs_ind, pred_ind)
    x_obs = x[obs_ind]
    x_pred = x[pred_ind]

    for i in range(hp_r['iterations']):
        # skip the first step, we can't calculate a mean from only one datapoint.
        bs_arr_modelworld[i, j] = ss_based_SNR(x_obs[:i+2], x_pred[:i+2])

bs_arr_modelworld = bs_arr_modelworld[:,~np.isinf(bs_arr_modelworld).any(0)]
# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.mean(bs_arr_modelworld, axis=1))
fig.show()

# Plot the SNR on the log scale
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.log(bs_arr_modelworld))
fig.show()

# Plot the SNR in decibel
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.transpose(10*np.log10(bs_arr_modelworld)))
fig.show()

# Here, values for the SNR can get incredibly large at the beginning.
# This is due to the little uncertainty in trajectories: they are basically the same until time step 15.

# This may be more suited for the perfect model scenario we have here,
# but I am not going to use it until I am sure of what the individual parts are
def var_based_SNR(obs, pred, inital_uncertainty):
    """
    The squared error based SNR, an estimator of the true expected SNR at perfect knowledge of parameters.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Dependent on sample size (decreases with sample size)
    """
    signal = np.dot(np.transpose(pred - np.mean(obs)), (pred - np.mean(obs)))
    noise = len(obs)*inital_uncertainty**2

    return signal/noise

#snr_list = []
#for i in range(len(x_obs)-1):
#    snr_list.append(var_based_SNR(x_obs[:i+2], x_pred[:i+2], theta_r2['sigma']))


# Let's try it differently and simpler. Take the mean of the forecast distribution as signal
# and the spread as noise. To get a distribution, we evolve the ensemble of trajectories.
# (Timothy delSole, 2004)

def tsnr(pred):
    # tSNR raw SNR or timeseries SNR: mean(timeseries) / std(timeseries)
    # tsnr increases with sample size (see sd).
    mu = np.mean(pred, axis=1)
    sd = np.std(pred, axis=1)#1/pred.shape[0]*np.sum(np.subtract(pred, mu)**2, axis=0)
    return mu**2/sd**2

snr_list = []
for i in range(hp_r['iterations']-1):
    snr_list.append(tsnr(x[:,:i+2]))

# Plot the SNR o
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.array(snr_list))
fig.show()

# This looks better.
# At least we see a convergence of this SNR after the expected number of time steps.
# HOWEVER: Here, we don't have an evaluation against anything: We're just evaluating the Forecast distribution against itself.
# Then, it doesn't make a difference, what setting I am in: the perfect model or real world scenario.
# A reference is required.

# We won't currently use it but are aware that there is a definition of a CNR as well.
# This is basically the same as the square-error-based SNR!
# Transfered, we have the model as the baseline and the mean as condition.
def cnr(x_c, x_b):
    """
    CNR - contrast to noise ratio: mean(condition-baseline) / std(baseline)
    time series contrast-to-noise ratio
    tsnr increases with sample size (see sd).
    """
    mu = np.mean(x_c) - np.mean(x_b)
    sd = 1/x.shape[0]*np.sum(np.subtract(x_b, mu)**2, axis=0)
    return mu/sd

# Now lets change world: From the Perfect Model Scenario, we move to the real world scenario.
# Doing so, we will use the real world model that is slightly more complex than the simple one.
# It contains an interaction with another species, that we will not consider (as if we weren't aware)

# Set hyperparameters.
hp_rm = {"iterations":30, "initial_size": (0.8, 0.9), "initial_uncertainty": 1e-1, "ensemble_size": 1}
# Set parameters
theta_rm = {'lambda_a': np.exp(1.8), 'lambda_b': np.exp(1.8), 'alpha':1, 'beta':10, 'gamma': 1, 'delta':10, 'sigma':0.1} # true parameter values (Petchey 2015)

ricker_multi = models.Ricker_Multi(uncertainties)
ricker_multi.set_parameters(theta_rm)
simu = ricker_multi.simulate(hp_rm, derive=False)
x_real = simu["ts"]
ricker_multi.visualise(np.transpose(x_real[:,:,0]), np.transpose(x_real[:,:,1]))

# We only use the data from one species, this is our real world data.
x_real = x_real[:,:,0].reshape(hp_rm['iterations'])

bs_samples = 50
bs_arr_dataworld = np.zeros((hp_r['iterations'], bs_samples))
for j in range(bs_samples):
    obs_ind, pred_ind = np.random.randint(hp_r['ensemble_size'], size=2)
    print(obs_ind, pred_ind)
    x_obs = x[obs_ind]
    x_pred = x[pred_ind]

    for i in range(hp_r['iterations']):
        # skip the first step, we can't calculate a mean from only one datapoint.
        bs_arr_dataworld[i, j] = ss_based_SNR(x_real[:i+2], x_pred[:i+2])

bs_arr_dataworld = bs_arr_dataworld[:,~np.isinf(bs_arr_dataworld).any(0)]
q1, q2 = np.quantile(bs_arr_dataworld, (0.1, 0.90), axis=1)
# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
ax.fill_between(np.arange(hp_r['iterations']), q1, q2, color="lightgrey", alpha=0.3)
plt.plot(np.mean(bs_arr_dataworld, axis=1))
fig.show()


# The task here is difficult through the chaotic behaviour of the model.
# In the perfect model scenario, what is it that we use as the variance of the noise?!
# If I use initial uncertainty, I will get a huge amplitude in the resulting SNR.

# Now the predictability limit as the limit where the SNR falls below some threshold.
# Here it makes a big difference if we use the mean or the median!
SNR_model = np.mean(bs_arr_modelworld, axis=1)
SNR_data = np.mean(bs_arr_dataworld, axis=1)
threshold = 1
pred_skill_model = np.argmax(SNR_model < threshold)
pred_skill_data = np.argmax(SNR_data < threshold)
print('Forecast horizon in the perfect model scenario: ', pred_skill_model)
print('Forecast horizon in the data world scenario: ', pred_skill_data)