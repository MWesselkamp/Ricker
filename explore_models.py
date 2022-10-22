# create simulator object
import simulations

sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(0.9)) # here we have to give init for both populations
x = sims.simulate()
mod = sims.ricker
derivative = mod.derive(x)

# create simulator object
import simulations
sims = simulations.Simulator(model_type="single-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=10,
                           ensemble_size=10,
                           initial_size=(0.9)) # here we have to give init for both populations
x_true = sims.simulate()
mod = sims.ricker
derivative = mod.derive(x_true)


# create simulator object
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=200,
                           ensemble_size=1,
                           initial_size=(0.9, 0.9)) # here we have to give init for both populations
x_true = sims.simulate()


# create simulator object
import simulations
sims = simulations.Simulator(model_type="multi-species",
                             simulation_regime="non-chaotic",
                             environment="non-exogeneous")
# Set hyperparameters. We'll simulate on a weekly resolution. Years is changed to weeks.
sims.hyper_parameters(simulated_years=200,
                           ensemble_size=1,
                           initial_size=(0.9, 0.9)) # here we have to give init for both populations
x_true = sims.simulate()


## Determine parameters for coexistence of two species
## Based on: May, R.M. Biological populations with non-overlapping generations: Stable points, stable cycles, and chaos. 1974

def D(a11, a12, a21, a22):
    return (a11*a22 - a12*a21)

def A(a11, K2, lambda2, N2_star, a22, K1, lambda1, N1_star):
    return ( (a11*K2/lambda2*N2_star) + (a22*K1/lambda1*N1_star))

D(1, 0.6, 1, 0.5)