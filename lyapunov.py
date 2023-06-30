import sympy as sp
import numpy as np

from simulations import simulate_temperature, generate_data

#==========#
# Example  #
#==========#

# Define the variables and parameters
x1, x2, x3 = sp.symbols('x1 x2 x3')
params = sp.symbols('param1 param2 param3')

# Define the non-linear dynamic equation
equation1 = params[0] * x1 + params[1] * x2**2 + params[2] * sp.sin(x3)
equation2 = x1 * x2 + x3**2
equation3 = params[0] * x1**2 + params[1] * sp.cos(x2) + params[2] * x3

# Define the system of equations
equations = [equation1, equation2, equation3]

# Create a list of variables
variables = [x1, x2, x3]

# Compute the Jacobian matrix
jacobian_matrix = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])

# Display the Jacobian matrix
jacobian_matrix


from numpy import linalg as LA
x = np.random.random()
Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])

#Now multiply a diagonal matrix by Q on one side and by Q.T on the other:
D = np.diag((-1,1))
LA.eigvals(D)
np.array([-1.,  1.])
A = np.dot(Q, D)
A = np.dot(A, Q.T)
LA.eigvals(A)
np.array([ 1., -1.])

#========================#
# Ricker: One-dimension  #
#========================#

timesteps = 50
growth_rate = 1.75
temp = simulate_temperature(timesteps=timesteps)
dyn_sim, dyn_obs = generate_data(timesteps = timesteps, growth_rate=growth_rate)
dyn_sim = dyn_sim.squeeze()
dyn_obs = dyn_obs.squeeze()

# accumulate Lyapunov over dime
lyapunov = 0
for i in range(timesteps):

    N = sp.symbols('N')
    params = sp.symbols('r g b c Temp')
    equation1 = N * sp.exp(params[0]*(1 - params[1]*N + params[2] * params[4] + params[3] * params[4] ** 2))
    variables = [N]
    jacobian_matrix = sp.Matrix([sp.diff(equation1, var) for var in variables])
    jacobian_matrix

    jacobian_func = sp.lambdify((N, params[0], params[1], params[2], params[3], params[4]), expr=jacobian_matrix)
    all_values = (dyn_sim[0], growth_rate, 0.96, 1.05, 1, temp[0])

    # Evaluate the Jacobian matrix at the given parameter values and state
    jacobian_evaluated = jacobian_func(*all_values)

    jacobian_evaluated = jacobian_evaluated.flatten()
    lyapunov += np.log(abs(jacobian_evaluated))

lyapunov/timesteps


#jacobian_evaluated = np.outer(jacobian_evaluated, jacobian_evaluated.T)
#eigenvalues = np.linalg.eigvals(jacobian_evaluated)

#========================#
# Ricker: Two-dimension  #
#========================#

jacobian_evaluated = np.ones((2,2))
growth_rate = (1.99, 1.89)
for i in range(timesteps):
    N1 ,N2 = sp.symbols('N1, N2')
    params = sp.symbols('r1 r2 a b c d e f Temp')
    equation1 = N1 * sp.exp(params[0]*(1 - params[2]*N1 - params[4]*N2+ params[6] * params[8] + params[7] * params[8] ** 2))
    equation2 = N2 * sp.exp(params[1]*(1 - params[3]*N2 - params[5]*N1 + params[6] * params[8] + params[7] * params[8] ** 2))

    variables = [N1, N2]
    equations = [equation1, equation2]
    jacobian_matrix = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    jacobian_matrix

    jacobian_func = sp.lambdify((N1, N2, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]), expr=jacobian_matrix)
    state_values = (dyn_obs[i,0], dyn_obs[i,1])
    param_values = (growth_rate[0], growth_rate[1], 1, 1, 0.0201, 0.02, 0.96, 1.05, temp[i])

    # Evaluate the Jacobian matrix at the given parameter values and state
    jacobian_evaluated *= jacobian_func(*state_values, *param_values)


dominant_eigenvalue = max(abs(np.linalg.eigvals(jacobian_evaluated)))
lyapunov = np.log(dominant_eigenvalue)/timesteps
lyapunov