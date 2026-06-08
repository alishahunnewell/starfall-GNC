"""Gravitational models for Phase 1: point-mass Earth.

This module provides the gravitational acceleration on a spacecraft at a
given inertial position. In Phase 1 I use a spherically symmetric Earth
(point-mass approximation); higher fidelity models (J2, full spherical
harmonics) can be swapped in later without changing the integrator.

References:
    Vallado, D. A., "Fundamentals of Astrodynamics and Applications," 4th ed.
    Curtis, H., "Orbital Mechanics for Engineering Students," 3rd ed.
"""

import numpy as np


# Earth gravitational parameter (m^3 / s^2).
# This is the product G * M_earth, measured directly from satellite tracking
# to far higher precision than G or M individually.
# Source: WGS84 / EGM2008
MU_EARTH = 3.986004418e14

# Earth equatorial radius in meters (WGS84).
# Used for converting between altitude and radius, and as a sanity check.
R_EARTH = 6378137.0


def point_mass_gravity(r):
    """Gravitational acceleration from a point-mass Earth.

    Implements Newton's law of universal gravitation in vector form:
        a = -mu * r / |r|^3

    The negative sign means the acceleration points from the spacecraft
    toward Earth's center (i.e., gravity pulls "down").

    Args:
        r: position vector from Earth's center to the spacecraft (m).
           Must be a numpy array of shape (3,).

    Returns:
        a: acceleration vector (m/s^2), shape (3,), pointing toward
           Earth's center.

    Raises:
        ValueError: if the position is too close to Earth's center, which
                    would cause numerical blow-up (1/r^3 explodes).
    """
    r_mag = np.linalg.norm(r)

    # Sanity check: if we're inside the Earth (or numerically at the center),
    # the point-mass model is unphysical and dividing by r^3 will give garbage.
    # In Phase 1 we never expect to integrate that close to the center anyway.
    if r_mag < 1.0:
        raise ValueError(
            f"Position too close to Earth center: |r| = {r_mag:.2f} m. "
            "Point-mass gravity is unphysical inside the Earth."
        )

    # The formula. Note we use r_mag**3 (not r_mag**2) because we're
    # multiplying by the full vector r, not just the unit vector.
    return -MU_EARTH * r / r_mag**3


def surface_gravity():
    """Magnitude of gravitational acceleration at Earth's mean surface (m/s^2).

    Useful for sanity checks and as a baseline for "what is g?"
    Returns approximately 9.798 m/s^2 (slightly lower than the conventional
    9.81 because the conventional value includes a centrifugal correction
    from Earth's rotation that we ignore here).
    """
    return MU_EARTH / R_EARTH**2