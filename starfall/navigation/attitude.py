"""Quaternion utilities for spacecraft attitude representation.

We use the SCALAR-FIRST, HAMILTON convention throughout:
    q = [w, x, y, z]
where w is the scalar part and (x, y, z) is the vector part.

A unit quaternion encodes a rotation that takes a vector from the
body frame to the inertial frame. That is, if r_body is a vector
in body-fixed coordinates, the same vector expressed in inertial
coordinates is r_inertial = R(q) @ r_body, where R(q) is the
rotation matrix derived from q.

A note on conventions:
    Different aerospace papers use scalar-first vs scalar-last
    (so [x,y,z,w]), and different multiplication conventions
    (Hamilton vs JPL). They are equally valid, but mixing them
    causes silent bugs. We pick one convention here and stick to it.
    NASA, Wikipedia, and most aerospace textbooks use scalar-first
    Hamilton. SciPy and some robotics packages use scalar-last.
    Be careful when interoperating.

References:
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control," Springer 2014, Ch. 2.
    Sola, "Quaternion kinematics for the error-state Kalman
        filter," arXiv:1711.02508
"""

import numpy as np


def quat_identity():
    """Return the identity quaternion [1, 0, 0, 0].

    This represents 'no rotation' — applying it to any vector leaves
    the vector unchanged. It's the rotational analogue of the number 1.
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q):
    """Return q divided by its magnitude, producing a unit quaternion.

    Numerical integration of dq/dt drifts the magnitude away from 1
    over time. Renormalizing every integration step (or every few)
    keeps it valid. Cheap operation, do it often.

    Args:
        q: quaternion as a 4-element array

    Returns:
        unit quaternion in the same direction
    """
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    """Hamilton-convention quaternion product: q1 ⊗ q2.

    The product represents "first apply q2, then apply q1." This is
    backwards from how you read it left-to-right, but matches how
    rotation matrices compose: R(q1 ⊗ q2) = R(q1) @ R(q2).

    The formula expanded:
        q1 = [w1, x1, y1, z1]
        q2 = [w2, x2, y2, z2]
        q1*q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,    # scalar part
                 w1*x2 + x1*w2 + y1*z2 - z1*y2,    # i component
                 w1*y2 - x1*z2 + y1*w2 + z1*x2,    # j component
                 w1*z2 + x1*y2 - y1*x2 + z1*w2]    # k component

    Args:
        q1, q2: 4-element quaternion arrays

    Returns:
        their Hamilton product as a 4-element array
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotation_matrix(q):
    """Convert a unit quaternion to its equivalent 3x3 rotation matrix.

    The resulting matrix R rotates vectors from body frame to inertial
    frame: r_inertial = R @ r_body.

    The closed-form expression is derived by working out what q ⊗ v ⊗ q*
    (where v is a vector treated as a pure quaternion [0, vx, vy, vz])
    does to v. The result is a rotation, and its matrix form is below.

    Args:
        q: 4-element unit quaternion [w, x, y, z]

    Returns:
        3x3 numpy array
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_kinematics(q, omega):
    """Time derivative of a quaternion given the body-frame angular velocity.

    This is the equation that connects "the spacecraft is rotating at
    angular velocity omega" to "the quaternion is changing at rate dq/dt."

    The formula:
        dq/dt = 0.5 * q ⊗ [0, omega_x, omega_y, omega_z]

    The factor of 1/2 comes from the half-angle convention in the
    quaternion definition. The [0, omega] construction treats angular
    velocity as a "pure" quaternion (zero scalar part).

    With this function in hand, the integrator can advance attitude
    through time just like it advances position: feed dq/dt back to RK4
    and let it do its thing.

    Args:
        q:     current attitude quaternion (4 elements)
        omega: angular velocity in BODY frame (3 elements, rad/s)

    Returns:
        dq/dt as a 4-element array
    """
    omega_as_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_as_quat)