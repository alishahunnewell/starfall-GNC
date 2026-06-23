"""Side-by-side comparison of PID and LQR controllers on identical hover problems.

Same vehicle, same initial conditions, same setpoint, same thrust limits.
Only difference: the controller. PID gains were hand-tuned; LQR gains were
derived by solving the Riccati equation given quadratic cost weights.

This script runs two complete simulations and overlays the results to make
the comparison visually direct.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from starfall.dynamics.integrators import integrate 
from starfall.dynamics.rigid_body import make_dynamics
from starfall.navigation.attitude import quat_identity 
from starfall.control.pid import PIDController
from starfall.control.lqr import solve_lqr, LQRController 

#parameters 
MASS = 1.0
INERTIA = np.eye(3) * 0.01
G_FLAT = 9.81 
T_MIN = 0.0
T_MAX = 4 * MASS * G_FLAT

#both controllers output thrust correction wrt gravity comp
#baseline and saturation limits on the correction are thus offset
U_MIN = T_MIN - MASS * G_FLAT
U_MAX = T_MAX - MASS * G_FLAT 

HOVER_ALTITUDE = 100.0
INITIAL_ALTITUDE = 80.0
T_END = 20.0
DT = 0.01 

#building PID controller (same method as before)
pid = PIDController(
    kp=6.0, ki=1.0, kd=4.5,
    setpoint=HOVER_ALTITUDE,
    u_min=U_MIN, u_max=U_MAX,
)

#building lqr controller, linearized dynamics around hover;
#   d/dt [delta_z]    = [0  1] [delta_z]   + [0   ] u
#        [delta_zdot]   [0  0] [delta_zdot]   [1/m]
# State cost: weight altitude error 10x more than velocity error.
# Control cost: small (thrust is cheap relative to error penalty).
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0],[1.0 / MASS]])
Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

K = solve_lqr(A, B, Q, R)
print(f"LQR gain K = {K}")

#lqr setpoint is "altitude error and velocity both zero at hover"
lqr = LQRController(
    K=K, 
    setpoint=np.array([0.0, 0.0]),
    u_min= U_MIN, u_max=U_MAX,
)

#initial state used by both simulations 
r0 = np.array([0.0, 0.0, INITIAL_ALTITUDE])
v0 = np.zeros(3)
q0 = quat_identity()
omega0 = np.zeros(3)
state0 = np.concatenate([r0, v0, q0, omega0])


def run_sim(controller_name, controller_update_fn):
    """Run one closed-loop simulation with a generic controller function.

    The controller_update_fn takes (t, state) and returns the thrust
    correction (a scalar). The simulation wraps this into the standard
    force/moment interface expected by the integrator.
    """
    thrust_history = []
    time_history = []

    def force_func(t, state):
        correction = controller_update_fn(t, state)
        # correction may come back as a length-1 array from LQR matrix math
        correction = float(np.asarray(correction).reshape(-1)[0])
        total_thrust = MASS * G_FLAT + correction
        thrust_history.append(total_thrust)
        time_history.append(t)
        return np.array([0.0, 0.0, -MASS * G_FLAT + total_thrust])

    def moment_func(t, state):
        return np.zeros(3)

    dynamics = make_dynamics(MASS, INERTIA, force_func, moment_func)
    ts, ys = integrate(dynamics, t_span=(0.0, T_END), y0=state0, dt=DT)
    return ts, ys, np.array(time_history), np.array(thrust_history)


# Wrap PID into the (t, state) -> correction interface
def pid_step(t, state):
    altitude = state[2]
    return pid.update(altitude, t)


# Wrap LQR similarly, extracting the 2-element regulator state from the
# full 13-element rigid body state.
def lqr_step(t, state):
    altitude = state[2]
    vertical_velocity = state[5]
    # LQR regulates altitude_error and velocity to zero, so we pass in
    # (altitude - setpoint, velocity).
    regulator_state = np.array([altitude - HOVER_ALTITUDE, vertical_velocity])
    return lqr.update(regulator_state)


# Run both simulations
ts_pid, ys_pid, th_pid_t, th_pid = run_sim("PID", pid_step)
ts_lqr, ys_lqr, th_lqr_t, th_lqr = run_sim("LQR", lqr_step)


def summarize(name, ts, ys):
    alts = ys[:, 2]
    vels = ys[:, 5]
    final_err = abs(alts[-1] - HOVER_ALTITUDE)
    max_err = np.max(np.abs(alts - HOVER_ALTITUDE))
    overshoot = max(0.0, alts.max() - HOVER_ALTITUDE)
    print(f"\n{name}:")
    print(f"  Final altitude:    {alts[-1]:.4f} m")
    print(f"  Steady-state err:  {final_err:.4f} m")
    print(f"  Peak overshoot:    {overshoot:.2f} m")
    print(f"  Max abs error:     {max_err:.2f} m")
    print(f"  Final velocity:    {vels[-1]:.4f} m/s")


summarize("PID", ts_pid, ys_pid)
summarize("LQR", ts_lqr, ys_lqr)

#plotting 
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

#ALTITUDE 
ax1 = axes[0]
ax1.plot(ts_pid, ys_pid[:, 2], color="cyan", linewidth=2.0, label="PID")
ax1.plot(ts_lqr, ys_lqr[:, 2], color="orange", linewidth=2.0, label="LQR")
ax1.axhline(HOVER_ALTITUDE, color="magenta", linestyle="--", linewidth=1.0, label="Setpoint")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Starfall GNC: PID vs LQR Hover Step Response")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.25)

# Vertical velocity
ax2 = axes[1]
ax2.plot(ts_pid, ys_pid[:, 5], color="cyan", linewidth=2.0, label="PID")
ax2.plot(ts_lqr, ys_lqr[:, 5], color="orange", linewidth=2.0, label="LQR")
ax2.axhline(0.0, color="white", linestyle=":", linewidth=0.5)
ax2.set_ylabel("Vertical velocity (m/s)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.25)

# Thrust
ax3 = axes[2]
ax3.plot(th_pid_t, th_pid, color="cyan", linewidth=1.0, alpha=0.6, label="PID")
ax3.plot(th_lqr_t, th_lqr, color="orange", linewidth=1.0, alpha=0.6, label="LQR")
ax3.axhline(MASS * G_FLAT, color="white", linestyle=":", linewidth=0.5,
            label=f"Gravity comp ({MASS*G_FLAT:.2f} N)")
ax3.axhline(T_MAX, color="red", linestyle=":", linewidth=0.5, label="Thrust limits")
ax3.axhline(T_MIN, color="red", linestyle=":", linewidth=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Thrust (N)")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.25)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "03_lqr_vs_pid_hover.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")

plt.show()