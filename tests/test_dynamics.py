"""Physics validation tests for the dynamics modules.

These tests check that the integrator and gravity model produce results
that match analytical solutions or well-known physical relationships.
If a future change breaks the physics, these tests fail loudly.

A note on tolerance:
    Numerical methods are never exact. We use pytest.approx with relative
    tolerances chosen so that correct physics passes but real bugs fail.
    For RK4 at moderate timesteps, ~0.1-1% relative tolerance is typical.
"""

import numpy as np
import pytest

from starfall.dynamics.integrators import rk4_step, integrate
from starfall.dynamics.gravity import (
    point_mass_gravity,
    surface_gravity,
    MU_EARTH,
    R_EARTH,
)


# Integrator tests

class TestIntegrator:
    """Tests for the RK4 integrator."""

    def test_constant_derivative_is_linear_in_time(self):
        """If dy/dt = constant, then y(t) should grow linearly.

        This is the simplest possible sanity check: a derivative function
        that returns a constant should produce y(t) = y0 + c*t.
        """
        def constant_rate(t, y):
            return np.array([2.0])  # dy/dt = 2

        ts, ys = integrate(constant_rate, (0, 5), np.array([10.0]), dt=0.1)

        # After 5 seconds at rate 2, we should have y = 10 + 2*5 = 20
        assert ys[-1, 0] == pytest.approx(20.0, rel=1e-10)

    def test_falling_object_matches_kinematics(self):
        """A falling object should match the schoolbook kinematic equations.

        State y = [position, velocity], with constant acceleration -g.
        At t=5s, position = 100 - 0.5*9.81*25 = -22.625
                velocity = -9.81 * 5 = -49.05
        """
        g = 9.81

        def falling(t, y):
            return np.array([y[1], -g])

        y0 = np.array([100.0, 0.0])
        ts, ys = integrate(falling, (0, 5), y0, dt=0.01)

        expected_position = 100.0 - 0.5 * g * 5**2
        expected_velocity = -g * 5

        assert ys[-1, 0] == pytest.approx(expected_position, rel=1e-6)
        assert ys[-1, 1] == pytest.approx(expected_velocity, rel=1e-6)

    def test_single_step_matches_full_integration(self):
        """rk4_step called manually in a loop should match integrate()."""
        def rate(t, y):
            return -y  # exponential decay: solution is y0 * exp(-t)

        y0 = np.array([1.0])
        dt = 0.01
        n_steps = 100

        # Manual loop
        y = y0.copy()
        t = 0.0
        for _ in range(n_steps):
            y = rk4_step(rate, t, y, dt)
            t += dt

        # Use integrate()
        ts, ys = integrate(rate, (0, n_steps * dt), y0, dt=dt)

        # They should agree to machine precision (same algorithm, same inputs)
        assert ys[-1, 0] == pytest.approx(y[0], rel=1e-12)



# Gravity tests


class TestGravity:
    """Tests for the point-mass gravity model."""

    def test_gravity_points_toward_center(self):
        """Gravity should always pull toward the origin."""
        # Pick a position 7000 km from Earth's center along +x.
        r = np.array([7_000_000.0, 0.0, 0.0])
        a = point_mass_gravity(r)

        # Acceleration should point in -x direction (toward center)
        assert a[0] < 0
        # And have no y or z component (since r had no y or z component)
        assert a[1] == pytest.approx(0.0, abs=1e-10)
        assert a[2] == pytest.approx(0.0, abs=1e-10)

    def test_gravity_magnitude_at_surface(self):
        """At Earth's surface, |g| should be ~9.798 m/s^2."""
        r = np.array([R_EARTH, 0.0, 0.0])
        a = point_mass_gravity(r)
        magnitude = np.linalg.norm(a)

        assert magnitude == pytest.approx(9.798, rel=1e-3)

    def test_gravity_obeys_inverse_square(self):
        """Doubling the distance should quarter the gravitational acceleration."""
        r1 = np.array([R_EARTH, 0.0, 0.0])
        r2 = np.array([2 * R_EARTH, 0.0, 0.0])

        a1 = np.linalg.norm(point_mass_gravity(r1))
        a2 = np.linalg.norm(point_mass_gravity(r2))

        # a2 should be a1 / 4 (inverse square law)
        assert a2 == pytest.approx(a1 / 4, rel=1e-10)

    def test_gravity_raises_at_center(self):
        """Calling at the origin should raise a ValueError, not return NaN/inf."""
        with pytest.raises(ValueError, match="too close to Earth center"):
            point_mass_gravity(np.array([0.0, 0.0, 0.0]))

    def test_surface_gravity_helper_matches_full_calculation(self):
        """The surface_gravity() helper should match the magnitude
        of point_mass_gravity at R_EARTH."""
        helper_value = surface_gravity()
        full_calculation = np.linalg.norm(
            point_mass_gravity(np.array([R_EARTH, 0.0, 0.0]))
        )
        assert helper_value == pytest.approx(full_calculation, rel=1e-12)



# Integrated tests (integrator + gravity together)


class TestDropFromAltitude:
    """End-to-end test: integrator + gravity, validated against
    a known analytical limit (constant-g approximation)."""

    def test_drop_from_100km_matches_analytical_within_1_percent(self):
        """Drop from 100 km altitude. Impact speed should match sqrt(2gh)
        to within 1% (the deviation is real physics, not numerical error,
        because gravity is weaker at altitude than at the surface)."""
        altitude = 100_000.0

        def dynamics(t, y):
            return np.concatenate([y[3:], point_mass_gravity(y[:3])])

        y0 = np.array([R_EARTH + altitude, 0, 0, 0, 0, 0], dtype=float)
        ts, ys = integrate(dynamics, (0, 200), y0, dt=0.1)

        # Find impact (altitude crosses zero)
        altitudes = np.linalg.norm(ys[:, :3], axis=1) - R_EARTH
        impact_idx = np.where(altitudes <= 0)[0]

        assert len(impact_idx) > 0, "Object didn't impact within 200 s"

        v_impact = np.linalg.norm(ys[impact_idx[0], 3:])
        v_analytical = np.sqrt(2 * surface_gravity() * altitude)

        # Within 1% — the deviation is from the real model being more accurate
        # than the constant-g analytical formula
        assert v_impact == pytest.approx(v_analytical, rel=0.01)