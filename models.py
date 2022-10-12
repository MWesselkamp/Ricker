import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

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

        self.uncertainties = uncertainties

    def uncertainty_properties(self, theta, sigma, initial_uncertainty):
        """
        Set true model parameters and precision for sampling under uncertainty.
        :param theta: dict. Model Parameters, i.e. r and sigma.
        :param precision:
        :return:
        """
        self.theta = theta
        self.sigma = sigma
        self.initial_uncertainty = initial_uncertainty

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

    def model_iterate(self, iterations, init, ex):

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
            timeseries = np.full((iterations,len(init)), init, dtype=np.float)
        else:
            timeseries = np.full(iterations, init, dtype=np.float)

        for i in range(1, iterations):

            # Exogeneous variable or not?
            if not ex is None:
                timeseries[i] = self.model(timeseries[i - 1], ex[i])
            else:
                timeseries[i] = self.model(timeseries[i - 1], ex)

            timeseries[i] = self.num.normal(timeseries[i], self.sigma)

        return timeseries

    def simulate(self, hp, ex = None):
        """
        Calls model class function iterate.
        :return: tuple. simulated timeseries and its derivative.
        """
        initial_size = hp["initial_size"]
        iterations = hp["iterations"]
        ensemble_size = hp["ensemble_size"]

        if not ensemble_size is None:

            timeseries_array = [None] * ensemble_size

            for n in range(ensemble_size):

                if type(initial_size) is np.ndarray:

                    initial_condition = initial_size[n]

                else:
                    initial_condition = self.num.normal(initial_size, self.initial_uncertainty)
                    # This is an artificial truncted normal distribution: Only use initial values above 1.
                    if type(initial_size) is tuple:
                        while any([i < 0 for i in (initial_size)]):
                            initial_condition = self.num.normal(initial_size, self.initial_uncertainty)
                    else:
                        while initial_condition < 0:
                            initial_condition = self.num.normal(initial_size, self.initial_uncertainty)

                timeseries = self.model_iterate(iterations, initial_condition, ex)
                timeseries_array[n] = timeseries

            self.simulations = {"ts":np.array(timeseries_array)}

        else:

            initial_condition = self.num.normal(initial_size, self.initial_uncertainty)
            timeseries = self.model_iterate(iterations, initial_condition)
            self.simulations = {"ts":timeseries}

        return self.simulations

    def derive_model(self):

        x = self.simulations["ts"]
        df_dN = []
        for j in range(x.shape[0]):
            df_dN_i = []
            for i in range(x.shape[1]):  # stepwise derivative
                N = x[j, i]
                N = torch.tensor(N, requires_grad=True)  # set requires_grad = True for computing the gradient
                if not self.ex is None:
                    dN = grad(self.model_torch(N, self.ex[i]), N)
                df_dN_i.append(torch.tensor(dN))  # turn tuple into tensor
            df_dN.append(torch.cat(df_dN_i).detach().numpy())
        df_dN = np.array(df_dN)

        return df_dN


    def visualise(self, x1, x2 = None):

        fig = plt.figure()
        ax = fig.add_subplot()
        if not x2 is None:
            plt.plot(x2, color="lightgrey")
        plt.plot(x1, color="blue")
        ax.set_xlabel('Time step t', size=14)
        ax.set_ylabel('Population size', size=14)
        fig.show()




class Ricker_Single(Model):

    def __init__(self, uncertainties, set_seed = True):
        """
        Initializes model as the Ricker model (Petchey 2015).
        """

        super(Ricker_Single, self).__init__(uncertainties, set_seed)

    def model(self, N, ex = None):
        """
        With or without stochasticity (Woods 2010).
        :param N: Population size at time step t.
        """
        return N * np.exp(self.theta['lambda']*(1- self.theta['alpha'] * N))

    def model_torch(self, N, ex = None):
        """
        Add numerical derivative.
        """
        return N * torch.exp(self.theta['lambda']*(1- self.theta['alpha'] * N))



class Ricker_Single_T(Model):

    def __init__(self, uncertainties, set_seed = True):
        """
        Initializes model as the Ricker model (Petchey 2015).
        """
        super(Ricker_Single_T, self).__init__(uncertainties, set_seed)

    def model(self, N, T):

        lambda_a = self.theta['ax'] + self.theta['bx'] * T + self.theta['cx'] * T**2

        return N * np.exp(lambda_a*(1 - self.theta['alpha'] * N))

    def model_torch(self, N, T):

        lambda_a = self.theta['ax'] + self.theta['bx'] * T + self.theta['cx'] * T ** 2

        return N * torch.exp(lambda_a * (1 - self.theta['alpha'] * N))


class Ricker_Multi(Model):

    def __init__(self, uncertainties, set_seed = True):

        super(Ricker_Multi, self).__init__(uncertainties, set_seed)

    def model(self, N, ex = None, fit = False):

        N_x, N_y = N[0], N[1]

        N_x_new =  N_x * np.exp(self.theta['lambda_a']*(1 - self.theta['alpha']*N_x - self.theta['beta']*N_y))
        N_y_new = N_y * np.exp(self.theta['lambda_b']*(1 - self.theta['gamma']*N_y - self.theta['delta']*N_x))

        if fit:
            return N_x_new
        else:
            return (N_x_new, N_y_new)

    def model_torch(self, N, ex=None):

        N_x, N_y = N[0], N[1]

        N_x_new = N_x * torch.exp(self.theta['lambda_a'] * (1 - self.theta['alpha'] * N_x - self.theta['beta'] * N_y))
        N_y_new = N_y * torch.exp(self.theta['lambda_b'] * (1 - self.theta['gamma'] * N_y - self.theta['delta'] * N_x))

        return (N_x_new, N_y_new)

class Ricker_Multi_T(Model):

    # Implement a temperature (and habitat size) dependent version of the Ricker Multimodel.
    # Mantzouni et al. 2010

    def __init__(self, uncertainties, set_seed = True):

        super(Ricker_Multi_T, self).__init__(uncertainties, set_seed)

    def model(self, N, T, fit=False):

        N_x, N_y = N[0], N[1]

        lambda_a = self.theta['ax'] + self.theta['bx'] * T + self.theta['cx'] * T**2
        lambda_b = self.theta['ay'] + self.theta['by'] * T + self.theta['cy'] * T**2

        N_x_new =  N_x * np.exp(lambda_a*(1- self.theta['alpha']*N_x - self.theta['beta']*N_y))
        N_y_new = N_y * np.exp(lambda_b*(1 - self.theta['gamma']*N_y - self.theta['delta']*N_x))

        if fit:
            return N_x_new
        else:
            return (N_x_new, N_y_new)

    def model_torch(self, N, T):

        N_x, N_y = N[0], N[1]

        lambda_a = self.theta['ax'] + self.theta['bx'] * T + self.theta['cx'] * T**2
        lambda_b = self.theta['ay'] + self.theta['by'] * T + self.theta['cy'] * T**2

        N_x_new =  N_x * torch.exp(lambda_a*(1- self.theta['alpha']*N_x - self.theta['beta']*N_y))
        N_y_new = N_y * torch.exp(lambda_b*(1 - self.theta['gamma']*N_y - self.theta['delta']*N_x))

        return (N_x_new, N_y_new)



