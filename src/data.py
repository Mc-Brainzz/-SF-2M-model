
import numpy as np
from scipy.integrate import solve_ivp

# Clean System: Standard Van der Pol Oscillator
def van_der_pol_ode(t, y, mu=1.0):
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]

# Biased System: Warped Van der Pol Oscillator
def biased_van_der_pol_ode(t, y, mu=2.5, k=1.5, f=0.5):
    # Incorrect physics and an external forcing term
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - k*x + f*np.cos(t)
    return [dxdt, dvdt]

# Low-quality Euler solver for the biased system
def euler_solver(func, y0, t_span, dt=0.1):
    t = np.arange(t_span[0], t_span[1], dt)
    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0
    for i in range(len(t) - 1):
        ys[i+1] = ys[i] + dt * np.array(func(t[i], ys[i]))
    return t, ys

def generate_data(n_samples=2000, t_eval_time=10.0):
    # Generate random initial conditions
    initial_conditions = (np.random.rand(n_samples, 2) - 0.5) * 6

    # Generate CLEAN data using a high-quality solver
    clean_data = np.array([solve_ivp(van_der_pol_ode, [0, t_eval_time], y0, dense_output=True).sol(t_eval_time) for y0 in initial_conditions])

    # Generate BIASED data using a low-quality solver and bad physics
    biased_data = np.array([euler_solver(biased_van_der_pol_ode, y0, [0, t_eval_time])[-1][-1] for y0 in initial_conditions])

    return biased_data, clean_data
