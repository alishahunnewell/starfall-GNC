"""simulated sensor models for testing navigation filter.

provides simple stochastic sensor classes that add realistic
noise to truth state values. IN real spacecraft these would be
replaced by actual hardware drivers. Each sensor has a measurement
equation, a noise model (guassian), a bias model, and a rate. 

this module is used by the phase 3 ekf demonstration script. it should
not be used for anything requiring physical realism beyond first 
order noise characterization. 
"""
import numpy as np 

class IMU:
    """simulated accelerometer with gaussian noise and constant bias.
    
    real IMUs have a random walk bias, temp dependence, scale factor
    errors, and misalignment. for phase 3 I modeled only gaussian
    white noise plus a constant bias, which captures dominant behavior
    in a short mission.
    """
    def __init__(self, noise_std=0.05, bias=0.0, rate_hz=100.0, seed=None):
        self.noise_std=noise_std
        self.bias = bias
        self.rate_hz = rate_hz
        self.rng = np.random.default_rng(seed)

    def measure(self, true_accel):
        noise = self.rng.normal(0.0, self.noise_std, size=np.shape(true_accel))
        return true_accel + self.bias + noise
    
class GPS:
    """simulated GPS position sensor.
    slow, noisy, but drift free (bias negligible for position senors 
    with satellite referenced solutions)
    """
    def __init__(self, noise_std=1.0, rate_hz=1.0, seed=None):
        self.noise_std=noise_std
        self.rate_hz = rate_hz
        self.rng = np.random.default_rng(seed)

    def measure(self, true_pos):
        noise = self.rng.normal(0.0, self.noise_std, size=np.shape(true_pos))
        return true_pos + noise