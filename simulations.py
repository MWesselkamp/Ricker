import numpy as np
from abc import ABC, abstractmethod
import torch
from torch.autograd import grad

class Model(ABC):

    def __init__(self):
        """
        Requires model object of type Ricker with corresponding class functions and attributes.
        :param mod: class. Model object.
        :param iterations: int. Time steps.
        :param obs_error: assume an observation error.
        """
        pass

    def parameters(self, theta, errors):
        """
        Set true model parameters and precision for sampling under uncertainty.
        :param theta: dict. Model Parameters, i.e. r and sigma.
        :param precision:
        :return:
        """
        self.theta = theta
        self.sigma = errors["sigma"]
        self.phi = errors["phi"]
        self.initial_uncertainty = errors["init_u"]

    def print_parameters(self):
        """
        Prints theta.
        """
        print("True parameter values:")
        try:
            for key, value in self.theta.items():
                print(key, value)
        except:
            print("Model parameters not set!")

    @abstractmethod
    def model(self, N):

        pass

    @abstractmethod
    def model_torch(self, N):

        pass

    def iterate(self, iterations, init, ex):

        """
        iterations: steps to integrate over.
        init: the set of initial conditions of state variable.
            Either float for single forecast or vector for ensemble forecast.
        ex: exogeneous variable.
        """
        self.ex = ex # required for derivative
        if type(init) is np.float64:
            init = init.item()

        # Simulate one trajectory or ensemble?
        if not type(init) is float:
            timeseries_true = np.full((iterations,len(init)), init, dtype=np.float)
            timeseries_obs = np.full((iterations, len(init)), init, dtype=np.float)
        else:
            timeseries_true = np.full(iterations, init, dtype=np.float)
            timeseries_obs = np.full(iterations, init, dtype=np.float)

        for i in range(1, iterations):

            # Exogeneous variable or not?
            if not ex is None:
                timeseries_true[i] = self.model(timeseries_true[i - 1], ex[i]) # true state
            else:
                timeseries_true[i] = self.model(timeseries_true[i - 1], ex) # true state

            timeseries_obs[i] = np.random.normal(timeseries_true[i], self.phi) # observed state

        return timeseries_obs, timeseries_true

    def simulate(self, iterations,initial_size, ensemble_size, ex = None):
        """
        Calls model class function iterate.
        :return: tuple. simulated timeseries and its derivative.
        """

        if not ensemble_size is None:

            timeseries_obs_array = [None] * ensemble_size
            timeseries_true_array = [None] * ensemble_size

            for n in range(ensemble_size):

                if type(initial_size) is np.ndarray:

                    initial_condition = initial_size[n]

                else:
                    initial_condition = np.random.normal(initial_size, self.initial_uncertainty)
                    # This is an artificial truncted normal distribution: Only use initial values above 1.
                    if type(initial_size) is tuple:
                        while any([i < 0 for i in (initial_size)]):
                            initial_condition = np.random.normal(initial_size, self.initial_uncertainty)
                    else:
                        while initial_condition < 0:
                            initial_condition = np.random.normal(initial_size, self.initial_uncertainty)

                timeseries_obs, timeseries_true = self.iterate(iterations, initial_condition, ex)
                timeseries_obs_array[n] = timeseries_obs
                timeseries_true_array[n] = timeseries_true


            self.simulations = {"ts_obs":np.array(timeseries_obs_array), "ts_true":np.array(timeseries_true_array)}

        else:

            initial_condition = np.random.normal(initial_size, self.initial_uncertainty)
            timeseries_obs, timeseries_true = self.iterate(iterations, initial_condition)
            self.simulations = {"ts_obs":timeseries_obs, "ts_true":timeseries_true}

        return self.simulations


    def derive(self, x):

        df_dN = []
        for j in range(x.shape[0]):
            df_dN_i = []
            for i in range(x.shape[1]):  # stepwise derivative
                N = x[j, i]
                N = torch.tensor(N, requires_grad=True)  # set requires_grad = True for computing the gradient
                if not self.ex is None:
                    dN = grad(self.model_torch(N, self.ex[i]), N)
                else:
                    dN = grad(self.model_torch(N, self.ex), N)
                df_dN_i.append(torch.tensor(dN))  # turn tuple into tensor
            df_dN.append(torch.cat(df_dN_i).detach().numpy())
        df_dN = np.array(df_dN)

        return df_dN



class Ricker_Single(Model):

    def __init__(self):
        """
        Initializes model as the Ricker model (Petchey 2015).
        """

        super(Ricker_Single, self).__init__()

    def model(self, N, ex = None):
        """
        With or without stochasticity (Woods 2010).
        :param N: Population size at time step t.
        """
        return N * np.exp(self.theta['alpha']*(1- self.theta['beta']* N)) + self.sigma*np.random.normal(0,1)

    def model_torch(self, N, ex = None):
        """
        Add numerical derivative.
        """
        return N * torch.exp(self.theta['alpha']*(1- self.theta['beta']* N)) + self.sigma*np.random.normal(0,1)



class Ricker_Single_T(Model):

    def __init__(self, set_seed = True):
        """
        Initializes model as the Ricker model (Petchey 2015).
        Environmental dependencies:
        https://academic.oup.com/icesjms/article/71/8/2307/2804451#139909426
        """
        super(Ricker_Single_T, self).__init__()

    def model(self, N, T):

        temp = self.theta['bx'] * T + self.theta['cx'] * T ** 2

        return N * np.exp(self.theta['alpha']*(1 - self.theta['beta']*N + temp)) + self.sigma*np.random.normal(0,1)

    def model_torch(self, N, T):

        lambda_a = self.theta['ax'] + self.theta['bx'] * T + self.theta['cx'] * T ** 2

        return N * torch.exp(lambda_a * (1 - N/self.theta['K'])) + self.sigma*np.random.normal(0,1)


class Ricker_Multi(Model):

    def __init__(self):

        super(Ricker_Multi, self).__init__()

    def model(self, N, ex = None, fit = False):
        """
        Based on: May, R.M. Biological populations with non-overlapping generations: Stable points, stable cycles, and chaos. 1974
        """
        N_x, N_y = N[0], N[1]

        N_x_new =  N_x * np.exp(self.theta['lambda1']*(1 - self.theta['alpha']*N_x - self.theta['beta']*N_y)) + self.sigma*np.random.normal(0,1)
        N_y_new = N_y * np.exp(self.theta['lambda2']*(1 - self.theta['gamma']*N_x - self.theta['delta']*N_y)) + self.sigma*np.random.normal(0,1)

        if fit:
            return N_x_new
        else:
            return (N_x_new, N_y_new)

    def model_torch(self, N, ex=None):

        N_x, N_y = N[0], N[1]

        N_x_new = N_x * torch.exp(self.theta['lambda1'] * (1 - self.theta['alpha'] * N_x - self.theta['beta'] * N_y)) + self.sigma*np.random.normal(0,1)
        N_y_new = N_y * torch.exp(self.theta['lambda2'] * (1 - self.theta['gamma'] * N_x - self.theta['delta'] * N_y)) + self.sigma*np.random.normal(0,1)

        return (N_x_new, N_y_new)

class Ricker_Multi_T(Model):

    # Implement a temperature (and habitat size) dependent version of the Ricker Multimodel.
    # Mantzouni et al. 2010
    # https://academic.oup.com/icesjms/article/71/8/2307/2804451#139909426

    def __init__(self):

        super(Ricker_Multi_T, self).__init__()

    def model(self, N, T, fit=False):

        N_x, N_y = N[0], N[1]

        temp_a = self.theta['bx'] * T + self.theta['cx'] * T**2
        temp_b = self.theta['by'] * T + self.theta['cy'] * T**2

        N_x_new =  N_x * np.exp(self.theta['ax']*(1- self.theta['alpha']*N_x - self.theta['beta']*N_y + temp_a)) + self.sigma*np.random.normal(0,1)
        N_y_new = N_y * np.exp(self.theta['ay']*(1 - self.theta['gamma']*N_x - self.theta['delta']*N_y + temp_b)) + self.sigma*np.random.normal(0,1)

        if fit:
            return N_x_new
        else:
            return (N_x_new, N_y_new)

    def model_torch(self, N, T):

        N_x, N_y = N[0], N[1]

        temp_a = self.theta['bx'] * T + self.theta['cx'] * T**2
        temp_b = self.theta['by'] * T + self.theta['cy'] * T**2

        N_x_new =  N_x * torch.exp(self.theta['ax']*(1 - self.theta['alpha']*N_x - self.theta['beta']*N_y + temp_a)) + self.sigma*np.random.normal(0,1)
        N_y_new = N_y * torch.exp(self.theta['ay'] *(1 - self.theta['gamma']*N_x - self.theta['delta']*N_y + temp_b)) + self.sigma*np.random.normal(0,1)

        return (N_x_new, N_y_new)


class Simulator:

    def __init__(self, model_type, growth_rate, environment, ensemble_size,initial_size):

        self.meta = {'model_type': model_type,
                     'growth_rate': growth_rate,
                     'environment': environment,
                     "initial_size": initial_size,
                    "ensemble_size": ensemble_size,
                    'hyper_parameters':None,
                    'model_parameters': None}

        self.growth_rate = growth_rate

    def simulate(self, sigma = 0, phi = 0, initial_uncertainty = 0, exogeneous=None):

        if not exogeneous is None:
            self.meta['iterations'] = len(exogeneous)
        else:
            self.meta['iterations'] = 365

        if (self.meta['model_type'] == "single-species") & (self.meta['environment']  == "non-exogeneous"):
            self.ricker = Ricker_Single()
            theta = {'alpha': self.growth_rate, 'beta': 1}


        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "non-exogeneous"):
            self.ricker = Ricker_Multi()
            theta = {'lambda1': self.growth_rate+0.0001,'alpha':1, 'beta':0.00006,
                    'lambda2': self.growth_rate, 'gamma':1, 'delta':0.00005}


        if (self.meta['model_type'] == "single-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = Ricker_Single_T()
            theta = {'alpha': self.growth_rate, 'beta': 1, 'bx': self.growth_rate + .01, 'cx': self.growth_rate + .1}

        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = Ricker_Multi_T()
            theta = {'alpha':1, 'beta':0.0201,
                    'gamma': 1, 'delta':0.02,
                    'ax': self.growth_rate+0.03, 'bx': self.growth_rate+ .01, 'cx': self.growth_rate+0.1,
                    'ay': self.growth_rate, 'by': self.growth_rate+ .01, 'cy':self.growth_rate+0.1}

        self.ricker.parameters(theta, {"sigma": sigma, "phi": phi, "init_u": initial_uncertainty})
        simu = self.ricker.simulate(self.meta["iterations"],
                                    self.meta["initial_size"],
                                    self.meta["ensemble_size"],
                                    ex=exogeneous)

        self.meta['theta'] = {'theta':theta}

        return simu

    def forecast(self, timesteps, observations = None, exogeneous = None):

        if not observations is None:
            analysis_distribution = np.random.normal(observations[:,-1], self.ricker.sigma, self.meta['ensemble_size'])
        else:
            analysis_distribution = self.ricker.simulations["ts"][:,-1]

        self.meta['initial_size'] = analysis_distribution
        self.meta['iterations'] = timesteps

        self.forecast_simulation = self.ricker.simulate(self.meta["iterations"],
                                    self.meta["initial_size"],
                                    self.meta["ensemble_size"],
                                    ex=exogeneous)


def simulate_temperature(timesteps, add_trend=False, add_noise=False):

    x = np.arange(timesteps)
    freq = timesteps / 365
    y = np.sin(2 * np.pi * freq * (x / timesteps))

    if add_trend:
        y = y + np.linspace(0, 0.1, timesteps)

    if add_noise:
        y = np.random.normal(y, 0.2)
    y = np.round(y, 4)

    return y

def generate_data(timesteps=50, growth_rate = 0.05,
                  sigma = 0.00, phi = 0.00, initial_uncertainty = 0.00,
                  doy_0 = 0, initial_size=1, ensemble_size = 10, environment = "exogeneous",
                  add_trend=False, add_noise=False):

    sims = Simulator(model_type="single-species",
                     environment=environment,
                     growth_rate=growth_rate,
                     ensemble_size=ensemble_size,
                     initial_size=initial_size)
    exogeneous = simulate_temperature(365+timesteps, add_trend = add_trend, add_noise = add_noise)
    exogeneous = exogeneous[365+doy_0:]
    xpreds = sims.simulate(sigma= sigma,phi= phi,initial_uncertainty=initial_uncertainty, exogeneous = exogeneous)['ts_obs']

    obs = Simulator(model_type="multi-species",
                    environment=environment,
                    growth_rate=growth_rate,
                    ensemble_size=1,
                    initial_size=(initial_size, initial_size))
    exogeneous = simulate_temperature(365+timesteps, add_trend = add_trend, add_noise = add_noise)
    exogeneous = exogeneous[365+doy_0:]
    xobs = obs.simulate(sigma= sigma,phi= phi,initial_uncertainty=initial_uncertainty, exogeneous = exogeneous)['ts_obs']

    return xpreds, xobs
