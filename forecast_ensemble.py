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

    def verify(self, ensemble, show=True):

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


    def plot(self, x, x_mean = None, transpose = False, log=False):

        fig = plt.figure()
        ax = fig.add_subplot()
        if log:
            x = np.log(x)
        if transpose:
            x = np.transpose(x)
        plt.plot(x, color="lightgrey")
        if not x_mean is None:
            plt.plot(np.mean(x, axis=x_mean), color="black", label="Ensemble mean")
        ax.set_xlabel("Time steps")
        ax.set_ylabel(self.metric)
        ax.legend(loc="upper left", prop={"size": 14})
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.show()


