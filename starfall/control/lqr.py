"""LQR (LINEAR QUADRATIC REGULATOR) optimal control
LQR is the optimal linear feedback controller for a linear dynamic system
under a quadratic cost function. Given:

    state-space dynamics:  dx/dt = A x + B u
    cost functional:       J = integral( x^T Q x + u^T R u ) dt

LQR finds the feedback gain K such that the control law u = -K x minimizes
J. This is provably the best linear controller possible for the given Q, R.

K is computed by solving the Continuous Algebraic Riccati Equation (CARE):

    A^T P + P A - P B R^-1 B^T P + Q = 0

for the positive-definite matrix P, then K = R^-1 B^T P.

Tuning intuition:
    larger Q   -> tighter tracking, more control effort
    larger R   -> less control effort, looser tracking
    Q and R are usually diagonal; off-diagonal terms couple variables.

References:
    Anderson and Moore, "Optimal Control: Linear Quadratic Methods,"
        Prentice Hall, 1990.
    Kalman, R.E., "Contributions to the Theory of Optimal Control,"
        Bol. Soc. Mat. Mexicana, 1960. (The original.)
"""

import numpy as np
from scipy.linalg import solve_continuous_are

def solve_lqr(A, B, Q, R):
    """Compute the LQR gain matrix K for the continuous time problem. 

    Solves the continuous algebraic Riccati equation and returns the 
    feedback gain K such that u = -K x is optimal under the cost 
    J= integral( x^T Q x + u^T R u ) dt.

    Args: 
        A: (n, n) state matrix 
        B: (n, m) input matrix 
        Q: (n, n) state cost matrix (symmetric, positive semi-definite)
        R: (m, m) control cost matrix (symmetric, positive definite)
    Returns:
        K: (m, n) feedback gain matrix
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    #scipy solves in one call 
    P = solve_continuous_are(A, B, Q, R)

    #now computing optimal gain using np.linalg rather  than explicitly computing
    K = np.linalg.solve(R, B.T @ P)
    return K

class LQRController:
    """ LQR state feedback controller with optional saturation.
    
    Wraps a precomputed gain matrix K with a clean update() interface
    that matches the PIDController API. The controller computes 
    u= -K (x-x_setpoint), then clips to the actuator limits. 
    Unlike PID, LQR has no internal state to track between updates:
    the gain is precomputed once at construction. 
    """
    def __init__(self, K, setpoint=None, u_min=-np.inf, u_max=np.inf):
        """
        Args:
            K: (m, n) LQR gain matrix from solve_lqr
            setpoint: (n,) desired state, or None for regulation to origin
            u_min, u_max: actuator saturation bounds (scalar or array)
        """
        self.K = np.asarray(K, dtype=float)
        self.setpoint = (np.zeros(self.K.shape[1]) if setpoint is None
                         else np.asarray(setpoint, dtype=float))
        self.u_min = u_min
        self.u_max = u_max

    def update(self, state, t=None):
        """Compute control output for the current state.

        Args:
            state: (n,) current state vector
            t: ignored (kept for API parity with PIDController)

        Returns:
            u: control output (saturated)
        """
        state = np.asarray(state, dtype=float)
        error = state - self.setpoint
        u_unsat = -self.K @ error
        return np.clip(u_unsat, self.u_min, self.u_max)
    