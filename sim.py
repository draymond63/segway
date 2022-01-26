from scipy.integrate import solve_ivp
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt

from controller import PID, Motor

class System():
    def __init__(self, g=9.81, length=1, mass=1, friction_coef=0.1, mtr_args=(0.9,), **pid_args):
        # System
        self.g = g
        self.L = length
        self.m = mass
        self.k = friction_coef
        # Controller
        self.ctrl = PID(**pid_args)
        self.mtr = Motor(*mtr_args)

    # State Space Representation
    def ode(self, t: float, y: list): # y=[theta, angular velocity]
        # System is time invariant, so it doesn't depend on t (timestep)
        theta, w = y
        # d0/dt
        dtheta_dt = w
        # From equations (in intro.ipynb)
        a = 3*self.g/(2*self.L)
        b = 3/(self.m*self.L)
        # Not actual forces but they are proportional to force
        Fg = a*np.sin(theta) # Gravity
        Fc = b*self.control(theta, t) # Control
        Ff = b*self.k*w # Friction
        # dw/dt
        dw_dt = Fg - Fc - Ff
        return dtheta_dt, dw_dt

    # PID controller shouldn't have access to any simulation variables
    # except theta, which will be measured
    def control(self, theta, t):
        signal = self.ctrl(theta, t)
        torque = self.mtr(signal, normalize=False)
        return torque

    def simulate(self, end, theta, w):
        sol = solve_ivp(self.ode, t_span=(0, end), y0=(theta, w), method='LSODA')
        thetas, ws = sol.y
        return sol.t, thetas, ws

    # Determine how fast the error is minimized, and whether the system is stable
    def solve(self, w_limit=10, max_duration=50, window=5):
        dur = max_duration # ! Initially test smaller duration before testing max dur
        for theta in np.linspace(0, np.pi):
            for w in np.linspace(0, w_limit):
                ts, thetas, ws = self.simulate(dur, theta, w)
                # ! Scrolling fft to determine when the peaks die out
                ...

if __name__ == '__main__':
    sys = System(Kp=5, Ki=0.1, Kd=0, update_rate=1/100)
    ts, thetas, ws = sys.simulate(end=70, theta=.1, w=.1)

    # Plot the numerical solution
    plt.grid()
    plt.plot(ts, thetas)
    plt.plot(ts, ws)
    plt.legend(('theta', 'angular velocity'))
    plt.show()