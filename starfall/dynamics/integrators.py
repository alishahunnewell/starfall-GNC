"""Numerical integrators for ordinary differential equations.

This module provides routines for advancing a state vector through time
given its derivative. The state derivative function must have the signature
f(t, y) -> dy/dt, where t is a scalar time and y is a numpy array.

For Phase 1 of Starfall, we use fixed-step RK4. This is sufficient for
ballistic and rigid-body dynamics over short trajectories. Phase 2 may
switch to adaptive-step methods (e.g., RK45) for longer simulations.
"""

import numpy as np


def rk4_step(f, t, y, dt):
    """Advance the state y by a single Runge-Kutta 4 step of size dt.

    RK4 estimates the derivative at four points within the interval [t, t+dt]:
        k1 at the start
        k2 at the midpoint, using k1 to step there
        k3 at the midpoint again, using k2 (more refined) to step there
        k4 at the endpoint, using k3 to step there

    The new state is the old state plus a weighted average of these slopes:
        y_new = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    The midpoints get double weight because they're more representative
    of the average behavior over the interval than the endpoints alone.

    Args:
        f:  callable f(t, y) -> dy/dt, where y is a numpy array
        t:  current time (seconds)
        y:  current state vector (numpy array, any shape)
        dt: timestep (seconds)

    Returns:
        y_new: state vector at time t + dt, same shape as y
    """
    k1 = f(t,          y)
    k2 = f(t + dt/2,   y + dt/2 * k1)
    k3 = f(t + dt/2,   y + dt/2 * k2)
    k4 = f(t + dt,     y + dt   * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def integrate(f, t_span, y0, dt):
    """Integrate an ODE from t_span[0] to t_span[1] with fixed step size dt.

    This is the high-level convenience wrapper around rk4_step. It builds
    the time array, allocates storage for the state history, and steps
    through the trajectory until the end time is reached.

    Args:
        f:      callable f(t, y) -> dy/dt
        t_span: (t_start, t_end) tuple of floats (seconds)
        y0:     initial state vector (numpy array)
        dt:     timestep (seconds)

    Returns:
        ts: 1-D array of time values, shape (N+1,)
        ys: 2-D array of states, shape (N+1, *y0.shape), where ys[i]
            is the state at time ts[i]
    """
    t_start, t_end = t_span

    # Figure out how many steps we need. np.ceil rounds up, so we always
    # cover the full time span (possibly slightly overshooting on the last step).
    n_steps = int(np.ceil((t_end - t_start) / dt))

    # Build the array of time samples. There are n_steps + 1 points because
    # we include both endpoints (t_start AND t_end).
    ts = np.linspace(t_start, t_end, n_steps + 1)

    # Allocate the output array for states. Shape is (n_steps + 1, *y0.shape),
    # which lets y0 be any dimensional array (a 1-D state vector, a matrix, etc.).
    ys = np.zeros((n_steps + 1, *np.shape(y0)))
    ys[0] = y0  # Initial condition

    # Step through the trajectory one RK4 step at a time.
    for i in range(n_steps):
        # Use the actual time difference between samples, not just dt.
        # This handles the case where the last step is slightly shorter.
        actual_dt = ts[i + 1] - ts[i]
        ys[i + 1] = rk4_step(f, ts[i], ys[i], actual_dt)

    return ts, ys