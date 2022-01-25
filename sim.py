from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class System():
    def __init__(self, g=9.81, length=1, mass=1, friction_coef=0.1, Kp=0, Ki=0, Kd=0):
        # System
        self.g = g
        self.L = length
        self.m = mass
        self.k = friction_coef
        # Controller
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_theta = 0
        self.prev_t = 0
        self.candidates = [0] * 2 # Buffer for prev values

    # State Space Representation
    def ode(self, y: list, t: float): # [theta, angular velocity]
        # System is time invariant, so it doesn't depend on t (timestep)
        theta, w = y
        # Update previous values if the simulation has moved on through time
        if self.candidates[0] != t:
            self.prev_t, self.prev_theta = self.candidates
        self.candidates = t, theta
        # d0/dt
        dtheta_dt = w
        # From equations (in intro.ipynb)
        a = 3*self.g/(2*self.L)
        b = 3/(self.m*self.L)
        # Not actual forces but they are proportional to force
        Fg = a*np.sin(theta) # Gravity
        Fc = b*self.PID(theta) # Control
        Ff = b*self.k*w # Friction
        # dw/dt
        dw_dt = Fg - Fc - Ff
        return dtheta_dt, dw_dt

    def PID(self, theta):
        w_estimate = (theta - self.prev_theta)
        # Calculate changes
        p = self.Kp*theta
        i = self.Ki*(theta + self.prev_theta)
        d = self.Kd*w_estimate
        return p + i + d

    def simulate(self, timesteps, theta, w):
        ys = odeint(self.ode, y0=(theta, w), t=timesteps)
        thetas, ws = ys.T
        return thetas, ws

ts = np.linspace(0, 5, 1000)
sys = System(Kp=10, Ki=10, Kd=0)

thetas, ws = sys.simulate(ts, theta=.1, w=.1)

# Plot the numerical solution
plt.grid()
plt.plot(ts, thetas)
plt.plot(ts, ws)
plt.legend(('theta', 'angular velocity'))
plt.show()

# plt.hist(sys.vals)
# plt.show()