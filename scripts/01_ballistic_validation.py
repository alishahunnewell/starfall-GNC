"""Ballistic trajectory of a non-rotating projectile in uniform gravity.

The simplest end-to-end validation case for the Starfall dynamics core. A
1 kg projectile is launched at 45 degrees from the horizontal with 100 m/s
of initial speed, under uniform 9.81 m/s^2 gravity, no atmosphere, no
thrust, no rotation. The simulated trajectory is compared against the
classical analytical solution:

    range    = v^2 * sin(2 theta) / g
    apex     = v^2 * sin^2(theta) / (2 g)
    flight_t = 2 v sin(theta) / g

Purpose:
    Validate that the full rigid_body_derivative + RK4 integration pipeline
    produces correct ballistic motion when reduced to a single force
    component, and generate the first plot committed to the project.

Note:
    Uses a flat-Earth uniform-g model so there's a closed-form solution to
    compare against. The point-mass Earth gravity model from gravity.py
    would give a slightly more accurate trajectory but no clean analytical
    reference. That's the next validation script.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from starfall.dynamics.integrators import integrate
from starfall.dynamics.rigid_body import make_dynamics
from starfall.navigation.attitude import quat_identity


# Vehicle and environment
MASS = 1.0                      # kg
INERTIA = np.eye(3) * 0.01      # kg m^2, arbitrary (no rotation)
G_FLAT = 9.81                   # m/s^2, uniform downward gravity

# Initial conditions
SPEED = 100.0                   # m/s
LAUNCH_ANGLE = np.deg2rad(45)

r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([
    SPEED * np.cos(LAUNCH_ANGLE),
    0.0,
    SPEED * np.sin(LAUNCH_ANGLE),
])
q0 = quat_identity()
omega0 = np.zeros(3)

state0 = np.concatenate([r0, v0, q0, omega0])


def force_func(t, state):
    """Uniform gravity in -z."""
    return np.array([0.0, 0.0, -MASS * G_FLAT])


def moment_func(t, state):
    """No moments; projectile doesn't tumble."""
    return np.zeros(3)


# Simulate
dynamics = make_dynamics(MASS, INERTIA, force_func, moment_func)
ts, ys = integrate(dynamics, t_span=(0.0, 25.0), y0=state0, dt=0.01)

# Trim at ground impact
altitudes = ys[:, 2]
ground_idx = np.where(altitudes < 0)[0]
if len(ground_idx) > 0:
    cutoff = ground_idx[0]
    ts = ts[:cutoff]
    ys = ys[:cutoff]

x_traj = ys[:, 0]
z_traj = ys[:, 2]


# Analytical comparison
flight_time_an = 2 * SPEED * np.sin(LAUNCH_ANGLE) / G_FLAT
range_an = SPEED**2 * np.sin(2 * LAUNCH_ANGLE) / G_FLAT
apex_an = SPEED**2 * np.sin(LAUNCH_ANGLE)**2 / (2 * G_FLAT)

t_analytical = np.linspace(0, flight_time_an, 200)
x_analytical = SPEED * np.cos(LAUNCH_ANGLE) * t_analytical
z_analytical = SPEED * np.sin(LAUNCH_ANGLE) * t_analytical - 0.5 * G_FLAT * t_analytical**2


# Report
err_range = abs(x_traj[-1] - range_an) / range_an * 100
err_apex = abs(z_traj.max() - apex_an) / apex_an * 100
err_time = abs(ts[-1] - flight_time_an) / flight_time_an * 100

print()
print("Phase 1 Ballistic Validation")
print(f"{'Quantity':<20}{'Simulated':>15}{'Analytical':>15}{'Error':>10}")
print(f"{'Range (m)':<20}{x_traj[-1]:>15.2f}{range_an:>15.2f}{err_range:>9.3f}%")
print(f"{'Apex altitude (m)':<20}{z_traj.max():>15.2f}{apex_an:>15.2f}{err_apex:>9.3f}%")
print(f"{'Flight time (s)':<20}{ts[-1]:>15.2f}{flight_time_an:>15.2f}{err_time:>9.3f}%")
print()


# Plot
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_analytical, z_analytical,
        color="magenta", linewidth=1.0, linestyle="--", label="Analytical (closed form)")
ax.plot(x_traj, z_traj,
        color="cyan", linewidth=2.0, label="Simulated (Starfall RK4 6-DOF)")

ax.scatter([x_traj[0]], [z_traj[0]], color="lime", s=80, zorder=5, label="Launch")
ax.scatter([x_traj[-1]], [z_traj[-1]], color="red", s=80, zorder=5, label="Impact")

ax.set_xlabel("Range (m)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"Starfall GNC: 45 deg ballistic launch at {SPEED:.0f} m/s")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_aspect("equal")

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "01_ballistic_validation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out_path}")

plt.show()