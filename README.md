# Forecast horizons of interspecific Ricker-type competition in silico 

Explores forecast horizons of dynamic behaviour for an extended time-discrete two-species coupled Ricker model, using different metrics, error types and scenarios. 

### Structure

models.py - contains the Ricker model classes to simulate from. Uncertainties considerable in the simulation:
                Process (sigma) and observation error. Parameter uncertainties (sd)

metrics.py - contains a range of proficiency metrics to select for FH computation.

forecast.py - contains dirty functions used in main to run full experiment.

main.py - run of this script creates version folder in results that contains experiment under four different simulation scenarios: determinstic/stochastic, chaotic/nonchaotic. 
Summary of simulated dynamics plot will look similar to:

![Examplary dynamics](results/version_240802_1102/dynamics_all.jpg)





