from logging import Logger

class PID():
    def __init__(self, Kp, Ki=0, Kd=0, update_rate=1/60, output_range=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accum = 0
        self.Hz = update_rate
        # Defaults output to 5V 10bit ADC output
        self.v_range = output_range if output_range else (0, 5)
        self.bits = 10
        self.signal = 0
        # Env tracking
        self.prev_theta = 0
        self.last_update = 0

    def norm(self, signal):
        max_output = (2**self.bits)
        if signal > max_output:
            Logger.warning(f"PID signal exceeds mapping, signal={signal}")
        v_min = self.v_range[0]
        v_max = self.v_range[-1]
        voltage = signal*v_max/max_output - v_min
        return voltage

    # Updates the output signal
    def update(self, theta: float, t: float):
        self.accum += theta
        # Calculate changes
        p = self.Kp*theta
        i = self.Ki*self.accum/t if t > 0 else 0
        d = self.Kd*(theta - self.prev_theta)
        # Save for next iteration
        self.prev_theta = theta
        return p + i + d

    def __call__(self, theta: float, t: float) -> float:
        if t - self.last_update > self.Hz:
            self.last_update = t
            self.signal = self.update(theta, t)
        return self.signal # ! Add normalization

class Motor():
    def __init__(self, maxT: float, v_range=(0, 5), lbin=False):
        self.maxT = maxT
        self.v_range = v_range

    # Scales voltage input to be between [0, 1]
    def norm(self, v: float):
        vmin, vmax = self.v_range
        if vmin <= v and v <= vmax:
            Logger.warning("Voltage supplied to motor is outside of the operating range")
        n = (v - vmin) / (vmax - vmin)
        return max(min(n, 1), 0)

    def __call__(self, volts: float, normalize=True):
        if normalize:
            return self.maxT * self.norm(volts)
        else:
            return self.maxT * volts