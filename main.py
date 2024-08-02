import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
import forecast as fc
import numpy as np
import os
import random
import pandas as pd

from utils import create_experiment_folder, create_scenario_folder
from visualisations import  plot_all_dynamics, plot_horizons, plot_horizons_relative_change
from forecast import create_experimental_data
from itertools import product
from horizons import Experiment

random.seed(42)
np.random.seed(42)


if __name__ == "__main__":

    new_experiment = True
    create_new_data = False

    directory_path = "results"

    if new_experiment:
        experiment_path = create_experiment_folder(directory_path)
    else:
        experiment_path = os.path.join(directory_path, 'version_230908_1004')

    # ========================#
    # Set simulation setting #
    # ========================#

    process = 'stochastic'
    scenario = 'nonchaotic'
    rows = ['deterministic', 'stochastic']
    cols = ['chaotic', 'nonchaotic']
    scenarios = list(product(rows, cols))
    scenario_folders  = [f'{s[1]}_{s[0]}' for s in scenarios]

    for s in scenarios:
        create_scenario_folder(experiment_path, f'{s[1]}_{s[0]}')

    if create_new_data:
        for s in scenarios:
            create_experimental_data(experiment_path, process=s[0], scenario= s[1], fit_model = False, compute_horizons = True)

    # Forecast load old experiment #
    observation, prediction, reference = fc.get_forecast_scenario(dir=os.path.join(directory_path, 'version_230908_1004'))

    plt.plot(prediction['chaotic_stochastic'].transpose())
    plt.plot(observation['chaotic_stochastic'])

    plt.plot(prediction['chaotic_deterministic'].transpose())
    plt.plot(observation['chaotic_deterministic'])


    # ========================#
    anomalies = {}

    for s in scenario_folders:

        obs = observation[s]
        preds = prediction[s]
        ref = reference[s]

        # set up experiment
        experiment = Experiment(obs, preds, ref, dir = os.path.join(experiment_path, s))
        experiment.compute_horizons()

        experiment.assemble_horizons(interval=1)
        all_horizons = experiment.horizons
        all_horizons = pd.DataFrame(all_horizons, columns=['correlation', 'anomaly', 'fstat', 'crps'],
                                    index=['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
        all_horizons.to_csv(os.path.join(experiment_path, s, 'horizons.csv'))
        plot_horizons(all_horizons, dir=os.path.join(experiment_path, s), show_upper=120, interval=1)
        plot_horizons(all_horizons, dir=os.path.join(experiment_path, s), show_upper=380, interval=1)

        experiment.assemble_horizons(interval=5)
        all_horizons = experiment.horizons
        all_horizons = pd.DataFrame(all_horizons, columns=['correlation', 'anomaly', 'fstat', 'crps'],
                                    index=['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
        all_horizons.to_csv(os.path.join(experiment_path, s, 'horizons.csv'))
        plot_horizons(all_horizons, dir=os.path.join(experiment_path, s), show_upper=120, interval=5)
        plot_horizons(all_horizons, dir=os.path.join(experiment_path, s), show_upper=380, interval=5)

        experiment.make_plots()

        anomalies[s] = [experiment.proficiencies['actual_ricker'][1],
                        experiment.proficiencies['intrinsic_ricker'][1],
                        experiment.proficiencies['actual_reference'][1],
                        experiment.proficiencies['intrinsic_reference'][1]]

        experiment.assemble_thresholds(full=False)
        all_thresholds = pd.DataFrame(experiment.thresholds, columns=['correlation', 'anomaly', 'fstat', 'crps'],
                                    index=['fha_ricker', 'fhp_ricker', 'fha_reference', 'fhp_reference', 'fsh'])
        all_thresholds.to_csv(os.path.join(experiment_path, s, 'thresholds.csv'))

    plot_all_dynamics(observation, prediction, reference, anomalies, dir=experiment_path)


    directory_path = 'results'
    experiment_path = os.path.join(directory_path, 'version_231207_1114')

    h_deterministic = pd.read_csv(os.path.join(experiment_path, 'chaotic_deterministic', 'horizons.csv'), index_col=0)
    h_stochastic = pd.read_csv(os.path.join(experiment_path, 'chaotic_stochastic', 'horizons.csv'), index_col=0)
    relative_change = h_deterministic -h_stochastic

    plot_horizons_relative_change(relative_change, dir=experiment_path, show_upper=120, interval=1)




