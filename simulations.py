import models
import numpy as np
import utils

class Simulator:

    def __init__(self, set_seed = False):

        self.uncertainties = {"parameters":False, "initial":True,"observation":True,"stoch":False}
        self.set_seed = set_seed

    def hyper_parameters(self,simulated_years,ensemble_size,initial_size):

        self.hp = {"iterations":simulated_years*52, "initial_size": initial_size,
                   "initial_uncertainty": 1e-1, "ensemble_size": ensemble_size}


    def simulation_parameters(self, regime, behaviour):

        if behaviour == "deterministic":
            self.uncertainties['observations'] == False

        self.regime = regime

        if self.regime == "chaotic":
            self.lam = 2.7
        elif self.regime == "non-chaotic":
            self.lam = 0.005

    def environment(self, environment, trend = False):

        self.environment = environment

        if self.environment == "exogeneous":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=trend, add_noise=True, show=False)
        elif self.environment == "non-exogeneous":
            self.T = None

    def model_type(self, type):

        self.type = type
        self.theta_upper = None

        if self.uncertainties['stoch'] == True:
            sigma = 0.3
        else:
            sigma = None

        if (self.type == "single-species") & (self.environment  == "non-exogeneous"):
            self.theta = {'lambda': self.lam, 'alpha': 1 / 1000, 'sigma': sigma}
            self.ricker = models.Ricker_Single(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (self.environment  == "non-exogeneous"):
            self.theta = {'lambda_a': self.lam, 'alpha':1/2000, 'beta':1/1900,
                          'lambda_b': self.lam, 'gamma': 1/2000, 'delta':1/1900,
                          'sigma':sigma}
            self.ricker = models.Ricker_Multi(self.uncertainties, self.set_seed)

        if (self.type == "single-species") & (self.environment == "exogeneous"):
            self.theta = { 'alpha': 1 / 1000, 'sigma': sigma}
            self.theta_upper = {'ax': self.lam, 'bx': .05, 'cx': .1}
            self.ricker = models.Ricker_Single_T(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (self.environment == "exogeneous"):
            self.theta = {'alpha':1/2000, 'beta':1/1900,
                          'gamma': 1/2000, 'delta':1/1900,
                          'sigma': sigma}
            self.theta_upper = {'ax': self.lam, 'bx': 0.05, 'cx': 0.1,
                                'ay': self.lam, 'by': 0.05, 'cy':0.1}
            self.ricker = models.Ricker_Multi_T(self.uncertainties, self.set_seed)


    def simulate(self, structured_samples = None):

        if not structured_samples is None:
            k = np.random.choice(np.arange(18, 50), structured_samples)
            r = np.random.normal(self.lam, 0.5, structured_samples)

            x = []
            for i in range(structured_samples):

                # Replace default carrying capacity and parameters by structured samples.
                self.hp['initial_size'] = k[i]
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

            if self.type == "single-species":
                self.ricker.visualise(np.transpose(x))
            else:
                self.ricker.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))

        return x



