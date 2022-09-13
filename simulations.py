import models
import numpy as np
import utils

class Simulator:

    def __init__(self, set_seed = False):

        self.uncertainties = {"parameters":False, "initial":True,"observation":True,"stoch":False}
        self.set_seed = set_seed

    def hyper_parameters(self,simulated_years,ensemble_size,initial_size):

        iterations = simulated_years*52
        self.hp = {"iterations":iterations, "initial_size": initial_size,
                   "initial_uncertainty": 1e-1, "ensemble_size": ensemble_size}


    def simulation_parameters(self, regime, behaviour):

        self.regime = regime

        if self.regime == "chaotic":
            self.lam = np.exp(2.2)
        elif self.regime == "non-chaotic":
            self.lam = np.exp(1.8)

        if behaviour == "deterministic":
            self.uncertainties['observations'] == False

    def choose_model(self, type, environment):

        self.type = type

        if environment == "exogeneous":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=False, add_noise=True)
        elif environment == "exogeneous-trend":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=True, add_noise=True)
        else:
            self.T = None

        self.theta_upper = None

        if (self.type == "single-species") & (environment  == "non-exogeneous"):
            self.theta = {'lambda': self.lam, 'alpha': 1 / 20, 'sigma': None}
            self.ricker = models.Ricker_Single(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (environment  == "non-exogeneous"):
            self.theta = {'lambda_a': self.lam, 'alpha':1/20, 'beta':30,
                          'lambda_b': self.lam, 'gamma': 1/20, 'delta':30,
                          'sigma':None}
            self.ricker = models.Ricker_Multi(self.uncertainties, self.set_seed)

        if (self.type == "single-species") & (environment == "exogeneous"):
            self.theta = { 'alpha': 1 / 20, 'sigma': None}
            self.theta_upper = {'ax': self.lam, 'bx': 2.5, 'cx': 2.2}
            self.ricker = models.Ricker_Single_T(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (environment == "exogeneous"):
            self.theta = {'alpha':1/20, 'beta':35,
                          'gamma': 1/20, 'delta':45,
                          'sigma': None}
            self.theta_upper = {'ax': self.lam, 'bx': 1.8, 'cx': 2.2,
                                'ay': self.lam, 'by': 1.0, 'cy':2.1}
            self.ricker = models.Ricker_Multi_T(self.uncertainties, self.set_seed)

    def simulate(self, structured_samples = None):

        if not structured_samples is None:
            k = np.random.choice(np.arange(18, 30), structured_samples)
            if self.regime == "non-chaotic":
                r = np.random.normal(self.lam, 0.1, structured_samples)

            x = []
            for i in range(structured_samples):
                self.theta['alpha'] = 1/k[i] # does not consider the second species. We're only looking at one anyways.
                if self.type == "single-species":
                    self.theta['lambda'] = np.exp(r[i])
                else:
                    self.theta_upper['ax'] = np.exp(r[i])
                self.ricker.set_parameters(self.theta, self.theta_upper)
                simu = self.ricker.simulate(self.hp, derive=False, ex=self.T)['ts']
                x.append(simu)

        else:
            self.ricker.set_parameters(self.theta, self.theta_upper)
            simu = self.ricker.simulate(self.hp, derive=False, ex=self.T)
            x = simu["ts"]

        return x



