import numpy as np
from abc import ABC, abstractmethod
import horizons
import metrics
import references

class ForecastEnsemble(ABC):

    def __init__(self, ensemble_predictions, reference):

        self.ensemble_predictions = ensemble_predictions
        self.reference_model = getattr(references, reference)
        self.meta = {'reference': reference,
                     'evaluation_style':None,
                     'metric':None,
                     'other': {'ensemble_index':None} }

    def verification_settings(self, metric, evaluation_style):

        self.metric_fun = getattr(metrics, metric)
        self.meta['evaluation_style'] = evaluation_style
        self.meta['metric'] = metric

class PerfectEnsemble(ForecastEnsemble):

    def __init__(self, ensemble_predictions, reference):

        super(PerfectEnsemble, self).__init__(ensemble_predictions, reference)

    def sample_ensemble_member(self):

        emsemble_index = np.random.randint(0, self.ensemble_predictions.shape[0], 1)
        control = self.ensemble_predictions[emsemble_index, :]
        ensemble_n = np.delete(self.ensemble_predictions, emsemble_index, axis=0)

        return control, ensemble_n, emsemble_index

    def accuracy(self):

        if self.meta['evaluation_style'] == "single":
            # self.ensemble index for plotting
            control, ensemble_n, emsemble_index = self.sample_ensemble_member()
            self.meta['other']['ensemble_index'] = emsemble_index

            reference_simulation = self.reference_model(control)

            self.accuracy_model = self.metric_fun(control, ensemble_n)
            self.accuracy_reference = self.metric_fun(control, reference_simulation)

        elif self.meta['evaluation_style'] == "bootstrap":

            bs = 100
            accuracy_model = []
            accuracy_reference = []
            reference_simulation = []

            for i in range(bs):

                control, ensemble_n, emsemble_index = self.sample_ensemble_member()

                reference_n = self.reference_model(control)
                reference_simulation.append(reference_n)

                accuracy_model.append(self.metric_fun(control, ensemble_n))
                accuracy_reference.append(self.metric_fun(control, reference_n))

            self.accuracy_model = np.array(accuracy_model)
            self.accuracy_reference = np.array(accuracy_reference)

        self.reference_simulation = reference_simulation

    def skill(self):

        return self.accuracy_model/self.accuracy_reference

    def skill_old(self):

        if self.meta['evaluation_style'] == "single":

            control, ensemble_n, emsemble_index = self.sample_ensemble_member()

            reference_simulation = self.reference_model(control)

            self.forecast_skill = self.metric_fun(reference_simulation, ensemble_n)

        elif self.meta['evaluation_style'] == "bootstrap":

            bs = 100
            forecast_skill = []
            reference_simulation = []

            for i in range(bs):

                control, ensemble_n, emsemble_index = self.sample_ensemble_member()

                reference_n = self.reference_model(control)

                reference_simulation.append(reference_n)
                forecast_skill.append(self.metric_fun(reference_n, ensemble_n))

            self.forecast_skill = np.array(forecast_skill)

        self.reference_simulation = reference_simulation

class PredictionEnsemble(ForecastEnsemble):

    def __init__(self, ensemble_predictions, observations, reference):

        self.observations = observations

        super(PredictionEnsemble, self).__init__(ensemble_predictions, reference)

    def accuracy(self):

        if self.meta['evaluation_style'] == "single":

            reference_n = self.reference_model(self.observations)

            self.accuracy_model = self.metric_fun(self.observations, self.ensemble_predictions)
            self.accuracy_reference = self.metric_fun(self.observations, reference_n)
            self.reference_simulation = reference_n

    def skill(self):

        self.forecast_skill = self.accuracy_model/self.accuracy_reference
        return self.forecast_skill

    def skill_old(self):

        if self.meta['evaluation_style'] == "single":

            reference_n = self.reference_model(self.observations, self.ensemble_predictions)

            self.reference_simulation = reference_n
            self.forecast_skill = self.metric_fun(reference_n, self.ensemble_predictions)

    def horizon(self, fh_type, threshold = None):

        if not threshold is None:

            horizon_fun = getattr(horizons, fh_type)
            fh_expectation, fh_dispersion, self.fh_matrix = horizon_fun(self.forecast_skill, threshold)
            return fh_expectation, fh_dispersion