from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as ProgressDisplay

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
    # except theta and time, which will be measured
    def control(self, theta, t):
        signal = self.ctrl(theta, t)
        torque = self.mtr(signal, normalize=False)
        return torque

    def simulate(self, duration, theta, w, **kwargs):
        self.ctrl.reset()
        sol = solve_ivp(self.ode, t_span=(0, duration), y0=(theta, w), method='LSODA', **kwargs)
        thetas, ws = sol.y
        return sol.t, thetas, ws

    # Determine how fast the error is minimized, and whether the system is stable
    def solve(self, w_limit=10, duration=10, atol=1e-2, pbar=False, plot=False) -> float:
        dt = 0.001 # Time increment
        ts = np.arange(0, duration, dt) # Time steps
        tests = [] # List of first windows to be considered stable for each simulation

        # Initial position errors
        y0 = np.linspace(0, np.pi/3, 5)
        y0 = ProgressDisplay(y0) if pbar else y0

        # Cycle through various initial conditions to see how long
        # it takes for the controller to stabilize the error
        for theta in y0:
            for w in np.linspace(-w_limit, w_limit, 5):
                # Run simulation
                thetas = self.simulate(duration, theta, w, t_eval=ts)[1]
                # Scrolling fft to determine when the peaks die out
                tests.append(thetas[-1] < atol and np.all(thetas < np.pi/2))
                # If this simulation failed
                if plot and not tests[-1]:
                    plt.plot(ts, thetas)
                    plt.show()
        return sum(tests) / len(tests)

if __name__ == '__main__':
    sys = System(Kp=10, Ki=0, Kd=0, friction_coef=1, update_rate=1/100)
    speed = sys.solve(pbar=True)
    print(speed)