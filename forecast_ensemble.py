import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import metrics
import references

class ForecastEnsemble(ABC):

    def __init__(self, reference, metric, evaluation_style):

        self.metric = metric
        self.reference = reference
        self.eval_style = evaluation_style
        self.metric_fun = getattr(metrics, self.metric)
        self.reference_fun = getattr(references, self.reference)

class PerfectEnsemble(ForecastEnsemble):

    def __init__(self, reference, metric, evaluation_style):

        super(PerfectEnsemble, self).__init__(reference, metric, evaluation_style)

    def verify(self, ensemble):

        if self.eval_style == "single":

            c = np.random.randint(0, ensemble.shape[0], 1)
            control = ensemble[c, :]
            ensemble_n = np.delete(ensemble, c, 0)

            reference_n = self.reference_fun(control)

            self.verification_model = self.metric_fun(control, ensemble_n)
            self.verification_reference = self.metric_fun(control, reference_n)

        elif self.eval_style == "bootstrap":

            bs = 100
            verification_model = []
            verification_reference = []

            for i in range(bs):

                c = np.random.randint(0, ensemble.shape[0], 1)
                control = ensemble[c, :]
                ensemble_n = np.delete(ensemble, c, 0)

                reference_n = self.reference_fun(control)

                verification_model.append(self.metric_fun(control, ensemble_n))
                verification_reference.append(self.metric_fun(control, reference_n))

            self.verification_model = np.array(verification_model)
            self.verification_reference = np.array(verification_reference)


class HindcastingEnsemble(ForecastEnsemble):

    def __init__(self, reference, metric, evaluation_style):

        super(HindcastingEnsemble, self).__init__(reference, metric, evaluation_style)

    def verify(self, ensemble, truth):

        if self.eval_style == "single":

            reference_n = self.reference_fun(truth)

            self.verification_model = self.metric_fun(truth, ensemble)
            self.verification_reference = self.metric_fun(truth, reference_n)


class PredictionEnsemble(ForecastEnsemble):

    def __init__(self, reference, metric, evaluation_style):

        super(PredictionEnsemble, self).__init__(reference, metric, evaluation_style)

    def verify(self, ensemble, observations):

        if self.eval_style == "single":

            reference_n = self.reference_fun(observations)
            # last one is the historic mean from all data we have so far
            reference_n = np.full(reference_n.shape, reference_n[:,-1])

            self.reference_simulation = reference_n
            self.verification_forecast = self.metric_fun(reference_n, ensemble)