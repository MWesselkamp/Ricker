import models
import numpy as np
import utils

class Simulator:

    def __init__(self, model_type, simulation_regime, environment,
                 set_seed = False, print = True):

        self.set_seed = set_seed
        self.meta = {'model_type': model_type,
                     'regime': simulation_regime,
                     'environment': environment,
                     'hyper_parameters':None,
                     'model_parameters': None}
        if print:
            print("SIMULATION UNDER THE FOLLOWING CONDITIONS:")
            print("Type of Ricker Model that will be used:   ", self.meta['model_type'])
            print("Simulation from Ricker in the following regime:  ", self.meta['regime'])
            print("Exogeneous impact on state variable considered?   ", self.meta['environment'])


    def hyper_parameters(self,simulated_years,ensemble_size,initial_size):

        self.hp = {"iterations":simulated_years*52, "initial_size": initial_size,
                   "ensemble_size": ensemble_size}
        self.meta['hp'] = self.hp

    def simulation_setup(self, sigma, phi, initial_uncertainty, theta):

        if self.meta['environment'] == "exogeneous":
            self.T = utils.simulate_T(self.hp['iterations'], add_trend=False, add_noise=True, show=False)
        elif self.meta['environment'] == "non-exogeneous":
            self.T = None

        if self.meta['regime'] == "chaotic":
            lam = 2.7
        elif self.meta['regime'] == "non-chaotic":
            lam = 0.05
        else:
            lam = self.meta['regime']

        if (self.meta['model_type'] == "single-species") & (self.meta['environment']  == "non-exogeneous"):
            self.ricker = models.Ricker_Single(self.set_seed)
            theta = {'lambda': lam, 'K': 1}


        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "non-exogeneous"):
            self.ricker = models.Ricker_Multi(self.set_seed)
            theta = {'lambda1': lam+0.0001, 'K1': 1, 'alpha':1, 'beta':0.00006,
                    'lambda2': lam, 'K2': 1, 'gamma':1, 'delta':0.00005}


        if (self.meta['model_type'] == "single-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = models.Ricker_Single_T(self.set_seed)
            theta = { 'K': 1, 'ax': lam, 'bx': .02, 'cx': .05}


        if (self.meta['model_type'] == "multi-species") & (self.meta['environment'] == "exogeneous"):
            self.ricker = models.Ricker_Multi_T(self.set_seed)
            theta = {'K1': 1, 'alpha':1, 'beta':0.00006,
                    'K2':1, 'gamma': 1, 'delta':0.00005,
                    'ax': lam+0.0001, 'bx': 0.02, 'cx': 0.05,
                    'ay': lam, 'by': 0.02, 'cy':0.05}


        self.meta['model_parameters']['theta'] = {'theta':theta}
        self.ricker.parameters(theta, {"sigma":sigma,"phi":phi, "init_u":initial_uncertainty})

    def simulate(self, pars = None, show=True):

        """
        A convenience function. Alternatively, extract ricker from simulation object and simulate.
        :param structured_samples:
        :return:
        """

        if pars is None:
            self.meta['model_parameters'] = {'theta': None,
                                            'sigma': 0.00,
                                            'phi': 0.0001,
                                            'initial_uncertainty': 1e-5}

        elif pars == "structured":
            pass

        else:
            self.meta['model_parameters'] = pars

        self.simulation_setup(sigma=self.meta['model_parameters']['sigma'],
                              phi=self.meta['model_parameters']['phi'],
                              initial_uncertainty=self.meta['model_parameters']['initial_uncertainty'],
                              theta=self.meta['model_parameters']['theta'])

        simu = self.ricker.simulate(self.hp, ex=self.T)
        x = simu["ts"]

        # For visualizing on the fly
        if show:
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
