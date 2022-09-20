import numpy as np
import simulations
import matplotlib.pyplot as plt
from proficiency_metrics import squared_error_SNR, raw_SNR


# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=30,
                           initial_size=20)
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('exogeneous', trend=False)
sims.model_type("single-species")
x = sims.simulate()


# we create the perfect model setting by treating one trajectory from the simulated ensemble as observation.
# we bootstrap this procedure to propagate uncertainty in initial conditions and thereby get a distribution of the SNR.
bs_samples = 100
bs_arr_modelworld = np.zeros((sims.hp['iterations'], bs_samples))
for j in range(bs_samples):
    obs_ind, pred_ind = np.random.randint(sims.hp['ensemble_size'], size=2)
    print(obs_ind, pred_ind)
    x_obs = x[obs_ind]
    x_pred = x[pred_ind]

    for i in range(sims.hp['iterations']):
        # skip the first step, we can't calculate a mean from only one datapoint.
        bs_arr_modelworld[i, j] = squared_error_SNR(x_obs[:i+2], x_pred[:i+2])

bs_arr_modelworld = bs_arr_modelworld[:,~np.isinf(bs_arr_modelworld).any(0)]
# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(bs_arr_modelworld, color="blue", alpha=0.3)
plt.plot(np.mean(bs_arr_modelworld, axis=1), color="black")
ax.set_xlabel('Time (generations)', size=14)
ax.set_ylabel('SS-SNR', size=14)
ax.title.set_text('Model world')
#plt.plot(bs_arr_modelworld)
fig.show()

# Plot the SNR on the log scale
fig = plt.figure()
ax = fig.add_subplot()
#plt.plot(np.mean(np.log(bs_arr_modelworld), axis=1))
plt.plot(np.log(bs_arr_modelworld), color="blue", alpha=0.3)
ax.set_xlabel('Time (generations)', size=14)
ax.set_ylabel('Log(SS-SNR)', size=14)
ax.title.set_text('Model world')
fig.show()

# Plot the SNR in decibel
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.transpose(10*np.log10(bs_arr_modelworld)), color="blue", alpha = 0.3)
ax.set_xlabel('Time (generations)', size=14)
ax.set_ylabel('SS-SNR [decibel]', size=14)
ax.title.set_text('Model world')
fig.show()

# Here, values for the SNR can get incredibly large at the beginning.
# This is due to the little uncertainty in trajectories: they are basically the same until time step 15.


# Let's try it differently and simpler. Take the mean of the forecast distribution as signal
# and the spread as noise. To get a distribution, we evolve the ensemble of trajectories.
# (Timothy delSole, 2004)

snr_list = []
for i in range(sims.hp['iterations']-1):
    snr_list.append(raw_SNR(x[:,:i+2]))

# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.array(snr_list), color="blue", alpha = 0.3)
ax.set_xlabel('Time (generations)', size=14)
ax.set_ylabel('Raw SNR', size=14)
ax.title.set_text('Model world')
fig.show()



# Now lets change world: From the Perfect Model Scenario, we move to the real world scenario.
# Doing so, we will use the real world model that is slightly more complex than the simple one.
# It contains an interaction with another species, that we will not consider (as if we weren't aware)

# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=30,
                           initial_size=(20, 20)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('exogeneous', trend=False)
sims.model_type("multi-species")
x_real = sims.simulate()

# We only use the data from one species, this is our real world data.
x_real = x_real[:,:,1]#.reshape(sims.hp['iterations'])

bs_samples = 100
bs_arr_dataworld = np.zeros((sims.hp['iterations'], bs_samples))
for j in range(bs_samples):
    obs_ind, pred_ind = np.random.randint(sims.hp['ensemble_size'], size=2)
    print(obs_ind, pred_ind)
    x_obs = x_real[obs_ind]
    x_pred = x[pred_ind]

    for i in range(sims.hp['iterations']):
        # skip the first step, we can't calculate a mean from only one datapoint.
        bs_arr_dataworld[i, j] = squared_error_SNR(x_obs[:i+2], x_pred[:i+2])

bs_arr_dataworld = bs_arr_dataworld[:,~np.isinf(bs_arr_dataworld).any(0)]
q1, q2 = np.quantile(bs_arr_dataworld, (0.1, 0.90), axis=1)
# Plot the SNR
fig = plt.figure()
ax = fig.add_subplot()
#ax.fill_between(np.arange(sims.hp['iterations']), q1, q2, color="lightgrey", alpha=0.3)
plt.plot(bs_arr_dataworld, color="blue", alpha = 0.3)
ax.set_xlabel('Time (generations)', size=14)
ax.set_ylabel('SS-SNR', size=14)
ax.title.set_text('Data world')
fig.show()



SNR_model = np.mean(bs_arr_modelworld, axis=1)
SNR_data = np.mean(bs_arr_dataworld, axis=1)
threshold = 1
pred_skill_model = np.argmax(SNR_model < threshold)
pred_skill_data = np.argmax(SNR_data < threshold)
print('Forecast horizon in the perfect model scenario: ', pred_skill_model)
print('Forecast horizon in the data world scenario: ', pred_skill_data)