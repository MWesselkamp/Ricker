import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Model(ABC):

    def __init__(self, uncertainties, set_seed = True):
        """
        Requires model object of type Ricker with corresponding class functions and attributes.
        :param mod: class. Model object.
        :param iterations: int. Time steps.
        :param obs_error: assume an observation error.
        """
        if set_seed:
            self.num = np.random.RandomState(100)
        else:
            self.num = np.random.RandomState()

        self.parameters = uncertainties["parameters"] # parameter error still missing!
        self.initial = uncertainties["initial"] # check
        self.obs_error = uncertainties["observation"] # check
        self.stoch = uncertainties["stoch"] # check

    def set_parameters(self, theta, theta_upper=None):
        """
        Set true model parameters and precision for sampling under uncertainty.
        :param theta: dict. Model Parameters, i.e. r and sigma.
        :param precision:
        :return:
        """
        self.theta = theta
        self.theta_upper = theta_upper

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

    def sample_parameters(self):
        """
        Sample parameters from normal distribution with mean and mean*precision.
        Not used currently.
        """
        pars = []
        for par, mean in self.theta.items():
            pars.append(self.num.normal(mean, mean * self.precision, 1)[0])
        return pars

    @abstractmethod
    def model(self, N, stoch=False):

        pass

    @abstractmethod
    def model_derivative(self, N):

        pass

    def model_iterate(self, iterations, init, ex, obs_error=False, stoch=False):

        """
        Based on Wood, 2010: Statistical inference for noisy nonlinear ecological dynamic systems.
        iterations: int.
        init: float. Initial Population size.
        obs_error: Add noise to simulated ts?
        """
        if not type(init) is float:
            timeseries = np.full((iterations,len(init)), init, dtype=np.float)
        else:
            timeseries = np.full(iterations, init, dtype=np.float)

        for i in range(1, iterations):
            if not ex is None:
                timeseries[i] = self.model(timeseries[i - 1], ex[i], stoch)
            else:
                timeseries[i] = self.model(timeseries[i - 1], ex, stoch)
            if obs_error:
                #timeseries[i] = self.num.poisson(timeseries[i])  # Adjust distribution! No poisson without phi
                timeseries[i] = self.num.normal(timeseries[i], 1)
        return timeseries

    def model_derive(self, iterations, init, timeseries):

        timeseries_derivative = np.full(iterations, self.model_derivative(init), dtype=np.float)
        for i in range(1, iterations):
            timeseries_derivative[i] = self.model_derivative(timeseries[i])

        return timeseries_derivative

    def simulate(self, hp, derive=True, ex = None):
        """
        Calls model class function iterate.
        :return: tuple. simulated timeseries and its derivative.
        """
        initial_size = hp["initial_size"]
        initial_uncertainty= hp["initial_uncertainty"]
        iterations = hp["iterations"]
        ensemble_size = hp["ensemble_size"]

        if not ensemble_size is None:
            timeseries_array = [None] * ensemble_size
            timeseries_derivative_array = [None] * ensemble_size

            for n in range(ensemble_size):

                if self.initial:
                    initial_condition = self.num.normal(initial_size, initial_uncertainty)
                    # This is an artificial truncted normal distribution: Only use initial values above 1.
                    if type(initial_size) is tuple:
                        while any([i < 0 for i in (initial_size)]):
                            initial_condition = self.num.normal(initial_size, initial_uncertainty)
                    else:
                        while initial_condition < 0:
                            initial_condition = self.num.normal(initial_size, initial_uncertainty)
                else:
                    initial_condition = initial_size

                timeseries= self.model_iterate(iterations, initial_condition, ex, obs_error = self.obs_error, stoch = self.stoch)
                timeseries_array[n] = timeseries

                if derive:
                    timeseries_derivative = self.model_derive(iterations, initial_condition, timeseries)
                    timeseries_derivative_array[n] = timeseries_derivative
            self.res = {"ts":np.array(timeseries_array), "ts_d":np.array(timeseries_derivative_array)}
            return self.res

        else:
            if self.initial: # only if inital conditions uncertainty considered
                initial_condition = self.num.normal(initial_size, initial_uncertainty)
            else:
                initial_condition = initial_size

            timeseries = self.model_iterate(iterations, initial_condition, obs_error=self.obs_error, stoch=self.stoch)
            if derive:
                timeseries_derivative = self.model_derive(iterations, initial_condition, timeseries)
            else:
                timeseries_derivative = None

            self.res = {"ts":timeseries, "ts_d":timeseries_derivative}
            return self.res

    def visualise(self, x1, x2 = None):

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(x1, color="blue")
        if not x2 is None:
            plt.plot(x2, color="lightgrey")
        ax.set_xlabel('Time step t', size=14)
        ax.set_ylabel('Population size', size=14)
        fig.show()




class Ricker_Single(Model):

    def __init__(self, uncertainties, set_seed = True):
        """
        Initializes model as the Ricker model (Petchey 2015).
        """

        super(Ricker_Single, self).__init__(uncertainties, set_seed)

    def model(self, N, ex = None, stoch=False):
        """
        With or without stochasticity (Woods 2010).
        :param N: Population size at time step t.
        """
        return N * np.exp(self.theta['lambda']*(1- self.theta['alpha'] * N))

    def model_derivative(self, N):
        """
        Add numerical derivative.
        """
        pass

class Ricker_Single_T(Model):

    def __init__(self, uncertainties, set_seed = True):
        """
        Initializes model as the Ricker model (Petchey 2015).
        """

        super(Ricker_Single_T, self).__init__(uncertainties, set_seed)

    def model(self, N, T, stoch = False):

        if stoch:
            T = self.num.normal(T, self.theta['sigma'])
        lambda_a = self.theta_upper['ax'] + self.theta_upper['bx'] * T + self.theta_upper['cx'] * T**2

        return N * np.exp(lambda_a*(1 - self.theta['alpha'] * N))

    def model_derivative(self, N):
        """
        Derivative of the Ricker at time step t.
        :param N: Population size at time step t.
        """
        pass



class Ricker_Multi(Model):

    def __init__(self, uncertainties, set_seed = True):

        super(Ricker_Multi, self).__init__(uncertainties, set_seed)

    def model(self, N, ex = None, stoch = False):

        N_x, N_y = N[0], N[1]

        N_x_new =  N_x * np.exp(self.theta['lambda_a']*(1 - self.theta['alpha']*N_x - self.theta['beta']*N_y))
        N_y_new = N_y * np.exp(self.theta['lambda_b']*(1 - self.theta['gamma']*N_y - self.theta['delta']*N_x))

        return (N_x_new, N_y_new)


    def model_derivative(self):

        pass


class Ricker_Multi_T(Model):

    # Implement a temperature (and habitat size) dependent version of the Ricker Multimodel.
    # Mantzouni et al. 2010

    def __init__(self, uncertainties, set_seed = True):

        super(Ricker_Multi_T, self).__init__(uncertainties, set_seed)

    def model(self, N, T, stoch = False):

        N_x, N_y = N[0], N[1]

        lambda_a = self.theta_upper['ax'] + self.theta_upper['bx'] * T + self.theta_upper['cx'] * T**2
        lambda_b = self.theta_upper['ay'] + self.theta_upper['by'] * T + self.theta_upper['cy'] * T**2

        N_x_new =  N_x * np.exp(lambda_a*(1- self.theta['alpha']*N_x - self.theta['beta']*N_y))
        N_y_new = N_y * np.exp(lambda_b*(1 - self.theta['gamma']*N_y - self.theta['delta']*N_x))

        return (N_x_new, N_y_new)

    def model_derivative(self):

        pass




