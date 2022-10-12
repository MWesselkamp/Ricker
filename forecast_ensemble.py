import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import metrics

class ForecastEnsemble(ABC):

    def __init__(self, reference, metric):

        self.reference = reference
        self.metric = metric
        self.metric_fun = getattr(metrics, self.metric)

class PerfectEnsemble(ForecastEnsemble):

    def __init__(self, reference, metric):

        super(PerfectEnsemble, self).__init__(reference, metric)

    def verify(self, ensemble, show=True):

        if self.reference == "control":

            c = np.random.randint(0, ensemble.shape[0], 1)
            control = ensemble[c,:]
            ensemble_n = np.delete(ensemble, c, 0)

            self.verification = self.metric_fun(control, ensemble_n)

        elif self.reference == "bootstrap":

            bs = 100
            verification = []
            for i in range(100):
                c = np.random.randint(0, ensemble.shape[0], 1)
                control = ensemble[c, :]
                ensemble_n = np.delete(ensemble, c, 0)
                verification.append(self.metric_fun(control, ensemble_n))

            self.verification = np.array(verification)


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


