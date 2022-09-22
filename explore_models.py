import simulations

# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(100, 100)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('exogeneous', trend=False)
sims.model_type("multi-species")
x_true = sims.simulate()

# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(100)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('exogeneous', trend=False)
sims.model_type("single-species")
x_true = sims.simulate()

# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(100, 100)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('non-exogeneous', trend=False)
sims.model_type("multi-species")
x_true = sims.simulate()

# create simulator object
sims = simulations.Simulator()
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(100)) # here we have to give init for both populations
sims.simulation_parameters(regime="non-chaotic", behaviour="stochastic")
sims.environment('non-exogeneous', trend=False)
sims.model_type("single-species")
x_true = sims.simulate()