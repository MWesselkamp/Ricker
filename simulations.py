import models
import numpy as np
import utils

class Simulator:

    def __init__(self, model_type, simulation_regime, environment,
                 uncertainties = {"parameters":False, "initial":True,"process":True},
                 set_seed = False):

        self.uncertainties = uncertainties

        self.set_seed = set_seed
        self.meta = {'model_type': model_type,
                     'regime': simulation_regime,
                     'environment': environment,
                     'hyper_parameters':None,
                     'uncertainty_parameters': None}

        print("SIMULATION UNDER THE FOLLOWING CONDITIONS:")
        print("Type of Ricker Model that will be used:   ", self.meta['model_type'])
        print("Simulation from Ricker in the following regime:  ", self.meta['regime'])
        print("Exogeneous impact on state variable considered?   ", self.meta['environment'])
        print("UNCERTAINTIES CONSIDERED:")
        print("Consider parameter uncertainty:      ", uncertainties['parameters'])
        print("Consider initial condition uncertainty:      ", uncertainties['initial'])
        print("Consider process uncertainty, i.e. NOT deterministic:    ", uncertainties['process'])

    def hyper_parameters(self,simulated_years,ensemble_size,initial_size):

        self.hp = {"iterations":simulated_years*52, "initial_size": initial_size,
                   "ensemble_size": ensemble_size}
        self.meta['hp'] = self.hp

    def simulation_setup(self, sigma = 0.5, initial_uncertainty=1e-1, pars = None):

        if self.meta['environment'] == "exogeneous":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=False, add_noise=True, show=False)
        elif self.meta['environment'] == "non-exogeneous":
            self.T = None

        if self.meta['regime'] == "chaotic":
            lam = 2.7
        elif self.meta['regime'] == "non-chaotic":
            lam = 0.005

        if (self.meta['model_type'] == "single-species") & (self.meta['environment']  == "non-exogeneous"):
            self.ricker = models.Ricker_Single(self.uncertainties, self.set_seed)
            theta = {'lambda': lam, 'alpha': 1 / 1000, 'sigma': 1}


        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "non-exogeneous"):
            self.ricker = models.Ricker_Multi(self.uncertainties, self.set_seed)
            theta = {'lambda_a': lam, 'alpha':1/2000, 'beta':1/1950,
                    'lambda_b': lam, 'gamma': 1/2000, 'delta':1/1955}


        if (self.meta['model_type'] == "single-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = models.Ricker_Single_T(self.uncertainties, self.set_seed)
            theta = { 'alpha': 1 / 1000, 'ax': lam, 'bx': .08, 'cx': .05}


        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = models.Ricker_Multi_T(self.uncertainties, self.set_seed)
            theta = {'alpha':1/2000, 'beta':1/1950,
                    'gamma': 1/2000, 'delta':1/1955,
                    'ax': lam, 'bx': 0.08, 'cx': 0.05,
                    'ay': lam, 'by': 0.08, 'cy':0.05}


        self.meta['uncertainty_parameters'] = {'theta':theta,
                                               'sigma':sigma,
                                               'initial_uncertainty':initial_uncertainty,
                                               'pars':pars}
        self.ricker.uncertainty_properties(theta, sigma, initial_uncertainty)

    def simulate(self, pars = "default", structured_samples = False):

        """
        A convenience function. Alternatively, extract ricker from simulation object and simulate.
        :param structured_samples:
        :return:
        """
        self.simulation_setup()

        if pars == "default":

            simu = self.ricker.simulate(self.hp, ex=self.T)
            x = simu["ts"]

        elif pars == "structured":

            pass

        if self.meta['model_type'] == "single-species":
            self.ricker.visualise(np.transpose(x))
        else:
            self.ricker.visualise(np.transpose(x[:,:,0]), np.transpose(x[:,:,1]))

        return x

    def forecast(self, years, observations = None):

        if not observations is None:
            analysis_distribution = self.ricker.num.normal(observations[:,-1], self.ricker.sigma, self.hp['ensemble_size'])
        else:
            analysis_distribution = self.ricker.simulations["ts"][:,-1]

        self.hp['initial_size'] = analysis_distribution
        self.hp['iterations'] = years*52

        T_pred = utils.simulate_T(self.hp['iterations'], add_trend=False, add_noise=True, show=False)

        self.forecast_simulation = self.ricker.simulate(self.hp, ex=T_pred)
