"""6-degree-of-freedom rigid body equations of motion.

State vector layout (13 elements):
    state[0:3]   - position in inertial frame (m)
    state[3:6]   - velocity in inertial frame (m/s)
    state[6:10]  - attitude quaternion [w, x, y, z], body-to-inertial
    state[10:13] - angular velocity in BODY frame (rad/s)

The function rigid_body_derivative is the heart of the file: given the
current state and the externally applied forces and moments, it returns
the 13-element time derivative that can be handed to the RK4 integrator.

References:
    Wie, B., "Space Vehicle Dynamics and Control," 2nd ed., AIAA 2008.
        Chapter 3 covers attitude kinematics, Chapter 5 covers Euler's
        equation.
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control," Ch. 2-3.
"""

import numpy as np

from starfall.navigation.attitude import (
    quat_normalize,
    quat_kinematics,
)


def rigid_body_derivative(t, state, mass, inertia, force_inertial, moment_body):
    """Time derivative of the 13-element rigid body state vector.

    This function is shape-compatible with the rk4_step integrator: it
    takes (t, state) plus any extra parameters and returns dstate/dt.

    The equations implemented (one per state group):

        d(position)/dt = velocity                                (kinematics)

        d(velocity)/dt = F_inertial / mass                       (Newton II)

        d(quaternion)/dt = 0.5 * q ⊗ [0, omega]                  (quat. kin.)

        d(omega)/dt = I^-1 (M_body - omega × I omega)            (Euler eq.)

    The last equation is Euler's rotation equation. The cross-product
    term is the "gyroscopic torque" — it's the reason a spinning gyroscope
    precesses, and the reason spinning rockets are unintuitive to control.

    Args:
        t: current time (s). Passed for consistency with the integrator
           API; this function doesn't use it explicitly, but force_func
           or moment_func wrappers might.
        state: 13-element state vector (see module docstring for layout)
        mass: scalar vehicle mass (kg)
        inertia: 3x3 inertia tensor in the BODY frame (kg·m^2). Must be
                 symmetric positive-definite.
        force_inertial: 3-element force vector in INERTIAL frame (N).
                        This is the sum of all external forces: gravity,
                        thrust (already rotated to inertial), aero, etc.
        moment_body: 3-element moment vector in BODY frame (N·m). Torques
                     are naturally expressed in the body frame because
                     the inertia tensor lives there.

    Returns:
        dstate/dt: 13-element derivative vector
    """
    r = state[0:3]    # position
    v = state[3:6]    # velocity
    q = state[6:10]   # attitude quaternion
    w = state[10:13]  # angular velocity (body frame)

   
   
    dr_dt = v

    # Translational dynamics: dv/dt = F / m  (Newton's 2nd law) ---
    # Force is in inertial frame and so is velocity, no transformation needed.
    dv_dt = force_inertial / mass

    # Rotational kinematics: dq/dt from omega 
    # Renormalize q before use to suppress numerical drift. Cheap insurance.
    # Note: this doesn't change the integrated state — the integrator gets
    # back a clean derivative regardless of small magnitude errors in q.
    q_unit = quat_normalize(q)
    dq_dt = quat_kinematics(q_unit, w)

    # Rotational dynamics: Euler's equation 
    # Iω is the angular momentum in the body frame. The cross product
    # ω × (Iω) gives the gyroscopic torque that arises when you express
    # rotation in a body-fixed (rotating) frame instead of inertial.
    # Solving I @ dω/dt = M - ω×Iω gives the angular acceleration.
    #
    # use np.linalg.solve rather than computing I^-1 and multiplying
    # because solve is numerically more stable and slightly faster.
    Iw = inertia @ w
    dw_dt = np.linalg.solve(inertia, moment_body - np.cross(w, Iw))

    # Reassemble into a single 13-element derivative vector.
    return np.concatenate([dr_dt, dv_dt, dq_dt, dw_dt])


def make_dynamics(mass, inertia, force_func, moment_func):
    """Build a closed-over dynamics function suitable for the RK4 integrator.

    The integrator expects a function with signature f(t, y) that takes
    just time and state. rigid_body_derivative takes those plus mass,
    inertia, and force/moment vectors. This helper "captures" those extra
    arguments inside a closure so the resulting f(t, y) matches the
    integrator's expected signature.

    Usage:
        def gravity_only_force(t, state):
            return mass * point_mass_gravity(state[:3])

        def no_moments(t, state):
            return np.zeros(3)

        f = make_dynamics(mass, inertia, gravity_only_force, no_moments)
        ts, ys = integrate(f, (0, 100), state0, dt=0.01)

    Args:
        mass: scalar vehicle mass (kg)
        inertia: 3x3 body-frame inertia tensor (kg·m^2)
        force_func: callable (t, state) -> 3-element inertial force (N)
        moment_func: callable (t, state) -> 3-element body moment (N·m)

    Returns:
        a function f(t, y) ready for the integrator
    """
    def f(t, state):
        F = force_func(t, state)
        M = moment_func(t, state)
        return rigid_body_derivative(t, state, mass, inertia, F, M)
    return f