import pandas as pd
import itertools
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as ProgressDisplay

from controller import PID, Motor

class System():
    def __init__(self, g=9.81, length=1, mass=1, friction_coef=0.1, wheel_radius=0.5, mtr_args=(0.9,), **pid_args):
        # System
        self.g = g
        self.L = length
        self.m = mass
        self.k = friction_coef
        self.r = wheel_radius
        self.A = 1/3 # 1/12 < A < 1/3 (for moment of inertia of a rod)
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

        # # From equations (in intro.ipynb)
        # a = self.g/(2*self.A*self.L)
        # b = 1/(self.m*self.A*self.L)
        # # Not actual forces but they are proportional to force
        # Fg = a*np.sin(theta) # Gravity
        # Fc = b*self.r*np.cos(theta)*self.control(theta, t) # Control
        # Ff = b*self.k*w # Friction

        # dw/dt
        dw_dt = Fg - Fc - Ff
        return dtheta_dt, dw_dt

    # PID controller shouldn't have access to any simulation variables
    # except theta and time, which will be measured
    def control(self, theta, t):
        signal = self.ctrl(theta, t)
        torque = self.mtr(signal, normalize=False)
        return torque

    # TODO: Ki's or slow update rates result in simulation warnings 👀
    def simulate(self, duration, theta, w, cap=False, t_eval=None, **kwargs):
        self.ctrl.reset()
        t_eval = t_eval if not isinstance(t_eval, type(None)) else np.linspace(0, duration, 1000 * duration) 
        sol = solve_ivp(self.ode, t_span=(0, duration), y0=(theta, w), method='LSODA', t_eval=t_eval, **kwargs)
        # Parse args
        thetas, ws = sol.y
        ts = sol.t
        # Cutoff simulation if the device hits the floor
        if cap:
            statuses = thetas >= np.pi/2
            if np.any(statuses):
                failure_index = next(i for i, s in enumerate(statuses) if s)
                thetas = thetas[:failure_index]
                ws = ws[:failure_index]
                ts = ts[:failure_index]
        return ts, thetas, ws

    # Determine how fast the error is minimized, and whether the system is stable
    def solve(self, w_limit=10, duration=10, atol=1e-2, pbar=False, plot=False) -> float:
        dt = 0.001 # Time increment
        ts = np.arange(0, duration, dt) # Time steps
        tests = [] # List of first windows to be considered stable for each simulation

        # Initial position errors (rad) and angular velocities (rad/s)
        y0 = np.linspace(0, np.pi/3, 5)
        y0 = ProgressDisplay(y0) if pbar else y0
        w0 = np.linspace(-w_limit, w_limit, 5)

        # Cycle through various initial conditions to see how long
        # it takes for the controller to stabilize the error
        for theta, w in itertools.product(y0, w0):
            # Run simulation
            thetas = self.simulate(duration, theta, w, t_eval=ts)[1]
            # Scrolling fft to determine when the peaks die out
            tests.append(thetas[-1] < atol and np.all(thetas < np.pi/2))
            # If this simulation failed
            if plot and not tests[-1]:
                plt.plot(ts, thetas)
                plt.show()
        return sum(tests) / len(tests)

def test_controller(iters=5):
    KPs = np.linspace(0, 100, iters)
    KDs = np.linspace(0, 100, iters)
    URs = np.linspace(50, 1000, iters) # Hz
    scores = []

    for Kp, Kd, update_rate in ProgressDisplay([*itertools.product(KPs, KDs, URs)]):
        sys = System(Kp=Kp, Kd=Kd, update_rate=update_rate)
        score = sys.solve()
        scores.append({
            'score': score,
            'Kp': Kp,
            'Kd': Kd,
            'UR': update_rate,
        })
    df = pd.DataFrame.from_dict(scores)
    print(df.sample(5))
    df.to_csv("scores-2.csv", index=False)

if __name__ == '__main__':
    # test_controller(iters=10)
    sys = System(Kp=10, Ki=0, Kd=0)
    ts, thetas, ws = sys.simulate(10, theta=1, w=0, cap=False)
    plt.grid()
    plt.plot(ts, thetas)
    plt.show()
