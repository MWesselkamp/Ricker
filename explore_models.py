import simulations

# create simulator object
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(100, 100)) # here we have to give init for both populations
x_true = sims.simulate()

# create simulator object
import simulations
sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(100)) # here we have to give init for both populations
x_true = sims.simulate()
mod = sims.ricker
derivative = mod.derive_model()

# create simulator object
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=100,
                           ensemble_size=30,
                           initial_size=(100, 100)) # here we have to give init for both populations
x_true = sims.simulate()

# create simulator object
import simulations

sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=2,
                           ensemble_size=10,
                           initial_size=(100)) # here we have to give init for both populations
x = sims.simulate()
mod = sims.ricker
derivative = mod.derive_model()