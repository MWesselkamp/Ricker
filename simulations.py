import models
import numpy as np
import utils

class Simulator:

    def __init__(self, model_type, simulation_regime, environment,
                 uncertainties = {"parameters":False, "initial":True,"process":True},
                 set_seed = False):

        self.uncertainties = uncertainties

        self.type = model_type
        self.regime = simulation_regime
        self.environment = environment

        self.set_seed = set_seed

        print("SIMULATION UNDER THE FOLLOWING CONDITIONS:")
        print("Type of Ricker Model that will be used:   ", self.type)
        print("Simulation from Ricker in the following regime:  ", self.regime)
        print("Exogeneous impact on state variable considered?   ", self.environment)
        print("UNCERTAINTIES CONSIDERED:")
        print("Consider parameter uncertainty:      ", uncertainties['parameters'])
        print("Consider initial condition uncertainty:      ", uncertainties['initial'])
        print("Consider process uncertainty, i.e. NOT deterministic:    ", uncertainties['process'])

    def hyper_parameters(self,simulated_years,ensemble_size,initial_size):

        self.hp = {"iterations":simulated_years*52, "initial_size": initial_size,
                   "initial_uncertainty": 1e-1, "ensemble_size": ensemble_size}

    def simulation_setup(self):

        if self.environment == "exogeneous":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=False, add_noise=True, show=False)
        elif self.environment == "non-exogeneous":
            self.T = None

        if self.regime == "chaotic":
            lam = 2.7
        elif self.regime == "non-chaotic":
            lam = 0.005

        theta_upper = None

        if (self.type == "single-species") & (self.environment  == "non-exogeneous"):
            theta = {'lambda': lam, 'alpha': 1 / 1000, 'sigma': 1}
            self.ricker = models.Ricker_Single(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (self.environment  == "non-exogeneous"):
            theta = {'lambda_a': lam, 'alpha':1/2000, 'beta':1/1950,
                    'lambda_b': lam, 'gamma': 1/2000, 'delta':1/1955,
                    'sigma':1}
            self.ricker = models.Ricker_Multi(self.uncertainties, self.set_seed)

        if (self.type == "single-species") & (self.environment == "exogeneous"):
            theta = { 'alpha': 1 / 1000, 'sigma': 1}
            theta_upper = {'ax': lam, 'bx': .08, 'cx': .05}
            self.ricker = models.Ricker_Single_T(self.uncertainties, self.set_seed)

        if (self.type == "multi-species") & (self.environment == "exogeneous"):
            theta = {'alpha':1/2000, 'beta':1/1950,
                          'gamma': 1/2000, 'delta':1/1955,
                          'sigma': 1}
            theta_upper = {'ax': lam, 'bx': 0.08, 'cx': 0.05,
                                'ay': lam, 'by': 0.08, 'cy':0.05}
            self.ricker = models.Ricker_Multi_T(self.uncertainties, self.set_seed)

        self.ricker.set_parameters(theta, theta_upper)

    def simulate(self, pars = "default", structured_samples = False):

        """
        A convenience function. Alternatively, extract ricker from simulation object and simulate.
        :param structured_samples:
        :return:
        """
        self.simulation_setup()

        if pars == "default":

            simu = self.ricker.simulate(self.hp, derive=False, ex=self.T)
            x = simu["ts"]

        elif pars == "structured":

            pass

        if self.type == "single-species":
            self.ricker.visualise(np.transpose(x))
        else:
            self.ricker.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))

        return x



