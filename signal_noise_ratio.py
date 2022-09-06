import numpy as np
import models
import matplotlib.pyplot as plt

uncertainties = {"parameters":False, "initial":True,"observation":False,"stoch":True}
# Set hyperparameters.
hp_r = {"iterations":30, "initial_size": 0.8, "initial_uncertainty": 1e-2, "ensemble_size": 25}
# Set parameters
theta_r2 = {'lambda':np.exp(1.8), 'alpha':1, 'sigma':0.1} # lambda = exp(r)

# Initialize model 2
ricker = models.Ricker_2(uncertainties)
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
    """
    EPE_mean = np.dot(np.transpose(obs - np.mean(obs)), (obs - np.mean(obs)))
    EPE_model = np.dot(np.transpose(obs - pred), (obs - pred))
    signal = (EPE_mean - EPE_model)
    noise = EPE_model

    return signal/noise

# we create the perfect model setting by using one trajectory from the simulated ensemble as observation.
obs_ind = np.random.choice(np.arange(hp_r['ensemble_size']))
x_obs = x[obs_ind]
x = np.delete(x, obs_ind, axis=0)
pred_ind = np.random.choice(np.arange(hp_r['ensemble_size']))
x_pred = x[pred_ind]
x = np.delete(x, pred_ind, axis=0)

snr_list = []
for i in range(len(x_obs)-1):
    #EPE_mean = np.dot(np.transpose(x_obs - np.mean(x_obs)), (x_obs - np.mean(x_obs)))
    #print(EPE_mean)
    #EPE_model = np.dot(np.transpose(x_obs - x_pred), (x_obs - x_pred))
    #print(EPE_model)
    snr_list.append(ss_based_SNR(x_obs[:i+2], x_pred[:i+2]))

# Plot the SNR on the log scale
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(snr_list)
fig.show()
# Plot the SNR in decibel
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.transpose(10*np.log10(snr_list)))
fig.show()

# Here, values for the SNR get incredibly large at the beginning.
# This is due to the little uncertainty in trajectories: they are basically the same until time step 15.
# The interpretation of the squared-error-based SNR is of limited help here.
# Lets try and implement the variance-based SNR, that is the true expected SNR at perfect knowledge of parameters.
# This may be more suited for the perfect model scenario we have here.

def var_based_SNR(obs, pred, inital_uncertainty):
    """
    The squared error based SNR, an estimator of the true expected SNR.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    """
    signal = np.dot(np.transpose(pred - np.mean(obs)), (pred - np.mean(obs)))
    noise = len(obs)*inital_uncertainty**2

    return signal/noise

snr_list = []
for i in range(len(x_obs)-1):
    #EPE_mean = np.dot(np.transpose(x_obs - np.mean(x_obs)), (x_obs - np.mean(x_obs)))
    #print(EPE_mean)
    #EPE_model = np.dot(np.transpose(x_obs - x_pred), (x_obs - x_pred))
    #print(EPE_model)
    snr_list.append(var_based_SNR(x_obs[:i+2], x_pred[:i+2], theta_r2['sigma']))

# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(snr_list)
fig.show()
# Plot the SNR on the log scale
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.log(snr_list))
fig.show()
# Plot the SNR in decibel
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.transpose(10*np.log10(snr_list)))
fig.show()

# This didn't really help.
# Let's try it differently and simpler. Take the mean of the forecast distribution as signal
# and the spread as noise. To get a distribution, we evolve the ensemble of trajectories.
# (Timothy delSole, 2004)

def tsnr(pred):
    # tSNR raw SNR or timeseries SNR: mean(timeseries) / std(timeseries)
    mu = np.mean(pred)
    sd = 1/pred.shape[0]*np.sum(np.subtract(pred, mu)**2, axis=0)
    return mu**2/sd**2

snr_list = []
for i in range(len(x_obs)-1):
    #EPE_mean = np.dot(np.transpose(x_obs - np.mean(x_obs)), (x_obs - np.mean(x_obs)))
    #print(EPE_mean)
    #EPE_model = np.dot(np.transpose(x_obs - x_pred), (x_obs - x_pred))
    #print(EPE_model)
    snr_list.append(tsnr(x[:,i+1]))

# Plot the SNR o
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(snr_list)
fig.show()

# This looks better.
# At least we see a convergence of this SNR after the expected number of time steps.
# HOWEVER: Here, we don't have an evaluation against anything: We're just evaluating the Forecast distribution against itself.
# Then, it doesn't make a difference, what setting I am in: the perfect model or real world scenario.
# A reference is required.

# We won't currently use it but are aware that there is a definition of a CNR as well.
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
hp_rm = {"iterations":30, "initial_size": (0.8, 0.9), "initial_uncertainty": 1e-5, "ensemble_size": 1}
# Set parameters
theta_rm = {'lambda_a': np.exp(2.9), 'lambda_b': np.exp(2.9), 'alpha':1, 'beta':0.1, 'gamma': 1, 'delta':0.1, 'sigma':None} # true parameter values (Petchey 2015)

ricker_multi = models.Ricker_Multi(uncertainties)
ricker_multi.set_parameters(theta_rm)
simu = ricker_multi.simulate(hp_rm, derive=False)
x_real = simu["ts"]
ricker_multi.visualise(np.transpose(x_real[:,:,0]), np.transpose(x_real[:,:,1]))

# We only use the data from one species, this is our real world data.
x_real = x_real[:,:,0].reshape(hp_rm['iterations'])
# Let's plot them together: They look very similar on the first 15 steps.
# But then the model starts more obviously diverging from the real world data
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(x_real)
plt.plot(x_pred)
fig.show()

snr_list = []
for i in range(len(x_real)-1):
    #EPE_mean = np.dot(np.transpose(x_obs - np.mean(x_obs)), (x_obs - np.mean(x_obs)))
    #print(EPE_mean)
    #EPE_model = np.dot(np.transpose(x_obs - x_pred), (x_obs - x_pred))
    #print(EPE_model)
    snr_list.append(ss_based_SNR(x_real[:i+1], x_pred[:i+1]))

# This time, the expectations are represented by the variance-based SNR trajectory.
# So what threshold would we pick here?
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(snr_list)
fig.show()

# The task here is difficult through the chaotic behaviour of the model.
# In the perfect model scenario, what is it that we use as the variance of the noise?!
# If I use initial uncertainty, I will get a huge amplitude in the resulting SNR.



