import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    return [dx, dy, dz]

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

p = (sigma, beta, rho)  # Parameters of the system

y0 = [1.0, 1.0, 1.0]  # Initial state of the system

t_span = (0.0, 40.0)
t = np.arange(0.0, 40.0, 0.01)

result_odeint = odeint(lorenz, y0, t, p, tfirst=True)
result_solve_ivp = solve_ivp(lorenz, t_span, y0, args=p)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(result_odeint[:, 0],
        result_odeint[:, 1],
        result_odeint[:, 2])
ax.set_title("odeint")

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot(result_solve_ivp.y[0, :],
        result_solve_ivp.y[1, :],
        result_solve_ivp.y[2, :])
ax.set_title("solve_ivp")

plt.show()