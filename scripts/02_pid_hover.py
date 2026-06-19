"""PID-controlled altitude hover, the first closed-loop simulation.

A 1 kg vehicle starts at rest 80 m above the ground. Gravity pulls it
down. A PID controller commands vertical thrust to hold the rocket at
the 80 m setpoint. The simulation runs for 20 seconds with a 10 ms
timestep, the same RK4 + 6-DOF rigid body pipeline as Phase 1.

The plot shows three things:
    1. Altitude vs. time (with setpoint overlaid for reference)
    2. Vertical velocity vs. time
    3. Commanded thrust vs. time (with the gravity-compensation baseline)

A well-tuned PID drives altitude smoothly to the setpoint with minimal
overshoot. A poorly tuned PID oscillates or saturates. The gains below
were tuned by hand to give a clean response; tune them yourself to see
how each gain affects the trajectory.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from starfall.dynamics.integrators import integrate
from starfall.dynamics.rigid_body import make_dynamics
from starfall.navigation.attitude import quat_identity
from starfall.control.pid import PIDController


# Vehicle and environment
MASS = 1.0
INERTIA = np.eye(3) * 0.01
G_FLAT = 9.81

# Thrust limits (Newtons). Max thrust = 4x weight, min thrust = 0.
# Real rocket engines can't push downward, so the lower bound is 0.
T_MIN = 0.0
T_MAX = 4 * MASS * G_FLAT

# Hover setpoint
HOVER_ALTITUDE = 100.0

# PID gains. Tuned by hand for clean step response.
KP = 6.0
KI = 1.0
KD = 4.5

# Build the controller. Its job is to output a thrust *correction* on top
# of the gravity-compensation baseline. The total commanded thrust is
# baseline + correction, clipped to [T_MIN, T_MAX].
controller = PIDController(
    kp=KP,
    ki=KI,
    kd=KD,
    setpoint=HOVER_ALTITUDE,
    u_min=T_MIN - MASS * G_FLAT,
    u_max=T_MAX - MASS * G_FLAT,
)

# Initial state: at rest, 100 m up, no rotation
r0 = np.array([0.0, 0.0, 80.0])
v0 = np.zeros(3)
q0 = quat_identity()
omega0 = np.zeros(3)
state0 = np.concatenate([r0, v0, q0, omega0])


# Storage to record thrust history for plotting
thrust_history = []
time_history = []


def force_func(t, state):
    """Compute net force on the vehicle.

    Force = gravity + thrust, both in the inertial frame. Thrust is purely
    vertical here (no gimbal in Phase 2A). The PID controller decides the
    thrust correction; we add the gravity-comp baseline to get total thrust.
    """
    altitude = state[2]
    correction = controller.update(altitude, t)
    total_thrust = MASS * G_FLAT + correction

    # Log for plotting later
    thrust_history.append(total_thrust)
    time_history.append(t)

    gravity_force = np.array([0.0, 0.0, -MASS * G_FLAT])
    thrust_force = np.array([0.0, 0.0, total_thrust])
    return gravity_force + thrust_force


def moment_func(t, state):
    """No moments yet. Phase 2B will add attitude control."""
    return np.zeros(3)


# Simulate
dynamics = make_dynamics(MASS, INERTIA, force_func, moment_func)
ts, ys = integrate(dynamics, t_span=(0.0, 20.0), y0=state0, dt=0.01)


# Extract trajectories
altitudes = ys[:, 2]
velocities = ys[:, 5]

# The thrust history was recorded inside force_func at every RK4 sub-step
# (4 per timestep), so it's about 4x longer than ts. Down-sample by taking
# the first entry per integration step. This is a small approximation
# fine for plotting.
thrust_history = np.array(thrust_history)
time_history = np.array(time_history)


# Report final state
print(f"Final altitude:  {altitudes[-1]:.4f} m   (setpoint: {HOVER_ALTITUDE} m)")
print(f"Final velocity:  {velocities[-1]:.4f} m/s")
print(f"Steady-state error: {abs(altitudes[-1] - HOVER_ALTITUDE):.4f} m")
print(f"Max altitude error: {np.max(np.abs(altitudes - HOVER_ALTITUDE)):.4f} m")


# Plot
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax1 = axes[0]
ax1.plot(ts, altitudes, color="cyan", linewidth=2.0, label="Actual altitude")
ax1.axhline(HOVER_ALTITUDE, color="magenta", linestyle="--", linewidth=1.0, label="Setpoint")
ax1.set_ylabel("Altitude (m)")
ax1.set_title(f"Starfall GNC: PID Altitude Hold "
              f"(Kp={KP}, Ki={KI}, Kd={KD})")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.25)

ax2 = axes[1]
ax2.plot(ts, velocities, color="lime", linewidth=2.0)
ax2.axhline(0, color="white", linestyle=":", linewidth=0.5)
ax2.set_ylabel("Vertical velocity (m/s)")
ax2.grid(True, alpha=0.25)

ax3 = axes[2]
ax3.plot(time_history, thrust_history, color="orange", linewidth=1.0, alpha=0.6)
ax3.axhline(MASS * G_FLAT, color="white", linestyle=":", linewidth=0.5,
            label=f"Gravity comp baseline ({MASS*G_FLAT:.2f} N)")
ax3.axhline(T_MAX, color="red", linestyle=":", linewidth=0.5, label=f"Thrust limits")
ax3.axhline(T_MIN, color="red", linestyle=":", linewidth=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Thrust (N)")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.25)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "02_pid_hover.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out_path}")

plt.show()