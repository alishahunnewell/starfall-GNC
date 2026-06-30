"""Extended Kalman Filter for spacecraft state estimation.

The EKF tracks both a state estimate and a covariance matrix describing 
the uncertainty in that estimate. at each timestep it does the following:

1)predicts; propagates state and covariance forward using dynamic model and 
accelerometer input. 
2) updates (when measurement is received); compare the predicted measurement 
against the actual measurement, then corrects the state estimate weighed by 
relative confidence.

This file implements a general EKF class. The specific dynamics and measurement
models are passed in as functions, so the same EKF can be utilized for altitude
estimation, attitude estimation, full 6-DOF state estimation and more.


References:
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control," Ch. 5-7.
    Simon, D., "Optimal State Estimation: Kalman, H-infinity, and
        Nonlinear Approaches," Wiley, 2006.
    Kalman, R. E., "A New Approach to Linear Filtering and Prediction
        Problems," Journal of Basic Engineering, 1960. (The original.)
"""

import numpy as np 

class ExtendedKalmanFilter:
    """generic extended kalman filter
    
    The user supplies the dynamics function f, the measurement function h,
    their Jacobians F and H, and the process and measurement noise 
    covariances Q and R. The EKF then handles predict and update steps. 
    """

    def __init__(self, x0, P0, f, F_jacobian, Q):
        """
        Args:
            x0: (n,) initial state estimate
            P0: (n, n) initial covariance matrix (positive definite)
            f: callable f(x, u, dt) -> next state, the dynamics propagator
            F_jacobian: callable F(x, u, dt) -> (n, n) Jacobian of f w.r.t. x
            Q: (n, n) process noise covariance matrix
        """
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.f = f
        self.F_jacobian = F_jacobian
        self.Q = np.asarray(Q, dtype=float)

    def predict(self, u, dt):
        """Propagate state and covariance forward by dt.

        x_new = f(x, u, dt)
        P_new = F P F^T + Q

        where F is the Jacobian of f evaluated at the current estimate.
        The Jacobian linearizes the dynamics; this is the "Extended"
        part of EKF (vs. standard Kalman which assumes linear dynamics).
        """
        F = self.F_jacobian(self.x, u, dt)
        self.x = self.f(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, h, H_jacobian, R):
        """Apply a measurement to refine the estimate.

        Args:
            z: (m,) measurement vector
            h: callable h(x) -> predicted measurement
            H_jacobian: callable H(x) -> (m, n) measurement Jacobian
            R: (m, m) measurement noise covariance

        Algorithm:
            innovation:    y = z - h(x)
            innovation cov: S = H P H^T + R
            Kalman gain:   K = P H^T S^-1
            state update:  x_new = x + K y
            cov update:    P_new = (I - K H) P
        """
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)

        H = H_jacobian(self.x)
        y = z - h(self.x)
        S = H @ self.P @ H.T + R

        # K = P H^T S^-1, but we use solve for numerical stability:
        # we want P H^T S^-1, equivalent to solving S^T (K^T) = (P H^T)^T = H P^T
        # Since S and P are symmetric: K^T = solve(S, H @ P), then K = K^T.T
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        