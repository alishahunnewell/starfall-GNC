"""PID (Proportional-Integral-Derivative) controller.

A PID controller produces a control output based on three weighted terms:
    proportional: how far off we are from the setpoint
    integral:     how long we've been off
    derivative:   how fast the error is changing

The control output is:
    u = Kp * error + Ki * integral(error) + Kd * d(error)/dt

For Starfall's altitude-hold case, "error" is the altitude difference
between the desired hover altitude and the current altitude. The output
is a thrust command that gets added to the gravity-compensating baseline.

References:
    Astrom & Murray, "Feedback Systems: An Introduction for Scientists
        and Engineers," Princeton, 2008. Chapter 11.
"""

import numpy as np


class PIDController:
    """A simple single-axis PID controller with anti-windup.

    Anti-windup means: if the controller hits a saturation limit (max
    thrust), we stop accumulating the integral term. Without this, the
    integral grows unboundedly during saturation and causes huge overshoot
    when the system finally responds. It's the single most common
    real-world PID failure mode and a thing that comes up in interviews.
    """

    def __init__(self, kp, ki, kd, setpoint=0.0, u_min=-np.inf, u_max=np.inf):
        """
        Args:
            kp, ki, kd: PID gains
            setpoint: desired value of the controlled variable
            u_min, u_max: output saturation limits (e.g., thrust bounds)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.u_min = u_min
        self.u_max = u_max

        # Internal state
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        """Clear integral and derivative history."""
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, measurement, t):
        """Compute the control output given a new measurement.

        Args:
            measurement: the current value of the controlled variable
            t: current time (s), used to compute dt

        Returns:
            u: control output (saturated to [u_min, u_max])
        """
        error = self.setpoint - measurement

        # Derivative term (skipped on first call when we have no history)
        if self.prev_error is None:
            derivative = 0.0
        else:
            dt = t - self.prev_time
            if dt <= 0:
                derivative = 0.0
            else:
                derivative = (error - self.prev_error) / dt

        # Tentative output before checking saturation
        u_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Saturate
        u = np.clip(u_unsat, self.u_min, self.u_max)

        # Anti-windup: only integrate if we are NOT saturated, or if the
        # error would push us back into the linear region
        if u == u_unsat:
            if self.prev_time is not None:
                dt = t - self.prev_time
                if dt > 0:
                    self.integral += error * dt

        # Update history
        self.prev_error = error
        self.prev_time = t

        return u
            
