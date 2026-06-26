"""Powered descent guidance via lossless convexification.

Solves the fuel-optimal landing problem for a planar (2D) rocket: given
an initial position, velocity, and vehicle parameters, find the thrust
history that lands the vehicle at the origin with zero velocity while
minimizing fuel use.

The problem is formulated as a convex optimization using the lossless
convexification trick (Acikmese and Ploen 2007). The non-convex thrust
magnitude constraint

    T_min <= |T| <= T_max

is replaced with the convex pair

    |T| <= gamma
    T_min <= gamma <= T_max

where gamma is a slack variable. The objective minimizes the integrated
slack, which proxies fuel consumption.

This is the same family of algorithms used by SpaceX for Falcon 9
first-stage landing burns (Blackmore 2013).

References:
    Acikmese, B. and Ploen, S. R., "Convex Programming Approach to
        Powered Descent Guidance for Mars Landing," Journal of Guidance,
        Control, and Dynamics, 2007.
    Blackmore, L., "Lossless Convexification of Control Problems with
        State and Input Constraints," AIAA GNC Conference, 2013.
"""

import numpy as np
import cvxpy as cp


def solve_powered_descent(
    r0,
    v0,
    mass,
    g_vec,
    T_min,
    T_max,
    t_final,
    N,
    glideslope_angle_deg=None,
):
    """Solve the planar powered descent guidance problem.

    Args:
        r0: (2,) initial position vector [downrange, altitude] (m)
        v0: (2,) initial velocity vector [horizontal, vertical] (m/s)
        mass: vehicle mass (kg). Treated as constant for simplicity;
              full LCvx handles mass burn via change of variables but
              adds complexity we skip in this v1.
        g_vec: (2,) gravity vector, e.g. np.array([0.0, -9.81])
        T_min: minimum thrust magnitude (N)
        T_max: maximum thrust magnitude (N)
        t_final: total descent duration (s)
        N: number of discretization timesteps
        glideslope_angle_deg: minimum angle above horizontal that the
                              vehicle must remain above. None disables
                              this constraint. Typical value: 30-70 deg.

    Returns:
        dict with keys 'r', 'v', 'T', 'gamma', 'ts', 'status'.
    """
    dt = t_final / N
    ts = np.linspace(0.0, t_final, N + 1)


    r = cp.Variable((N + 1, 2))           # position at each step
    v = cp.Variable((N + 1, 2))           # velocity at each step
    T = cp.Variable((N, 2))               # thrust between steps
    gamma = cp.Variable(N, nonneg=True)   # slack variable

    constraints = []

    # initial conditions
    constraints.append(r[0] == r0)
    constraints.append(v[0] == v0)

    constraints.append(r[N] == np.zeros(2))
    constraints.append(v[N] == np.zeros(2))

    
    for k in range(N):
        constraints.append(r[k + 1] == r[k] + dt * v[k])
        constraints.append(v[k + 1] == v[k] + dt * (T[k] / mass + g_vec))

   
    for k in range(N):
        constraints.append(cp.norm(T[k], 2) <= gamma[k])
    constraints.append(gamma >= T_min)
    constraints.append(gamma <= T_max)

    
    if glideslope_angle_deg is not None:
        tan_angle = np.tan(np.deg2rad(glideslope_angle_deg))
        for k in range(N + 1):
            constraints.append(r[k, 1] >= tan_angle * cp.abs(r[k, 0]))

    # want to minimize integrated slack (proxy for fuel use).
    objective = cp.Minimize(cp.sum(gamma) * dt)

    # building
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return {
        "r": r.value,
        "v": v.value,
        "T": T.value,
        "gamma": gamma.value,
        "ts": ts,
        "status": problem.status,
    }