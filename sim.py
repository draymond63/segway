from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class System():
    def __init__(self, g=9.81, length=1, mass=1, Kp=0, Ki=0, Kd=0):
        # System
        self.g = g
        self.L = length
        self.m = mass
        # Controller
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_theta = 0
        self.prev_t = 0
        self.candidates = [0] * 2 # Buffer for prev values
        self.vals = []

    # State Space Representation
    def ode(self, y: list, t: float): # [theta, angular velocity]
        # System is time invariant, so it doesn't depend on t (timestep)
        theta, w = y
        # d0/dt
        dtheta = w
        # Not actual forces but they are proportional to force
        Fg = 3*self.g/(2*self.L)*np.sin(theta)
        Fc = 3/(self.m*self.L)*self.PID(theta, w, t)
        # dw/dt
        dw = Fg - Fc
        return dtheta, dw

    def PID(self, theta, w, t):
        # Update previous values if the simulation has moved on through time
        if self.candidates[0] != t:
            self.prev_t, self.prev_theta = self.candidates
        self.candidates = t, theta
        dt = t - self.prev_t

        # In case simulation repeats timesteps
        if dt <= 0:
            return 0
        w_estimate = (theta - self.prev_theta)/dt
        self.vals.append(w - w_estimate)

        # Calculate changes
        p = self.Kp*theta
        i = self.Ki*(theta + self.prev_theta)/dt
        d = self.Kd*w_estimate
        return p + i + d

    def simulate(self, timesteps, theta, w):
        ys = odeint(self.ode, y0=(theta, w), t=timesteps)
        thetas, ws = ys.T
        return thetas, ws

ts = np.linspace(0,2.7,1000)
sys = System(Kp=10, Ki=0, Kd=1)

thetas, ws = sys.simulate(ts, theta=.1, w=.1)

# Plot the numerical solution
plt.grid()
plt.plot(ts, thetas)
plt.plot(ts, ws)
plt.legend(('theta', 'angular velocity'))
plt.show()

# plt.hist(sys.vals)
# plt.show()