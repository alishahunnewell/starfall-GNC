"""Powered descent guidance trajectory visualization.

Runs the lossless-convexification solver to compute a fuel-optimal landing
trajectory for a planar rocket, then plots the result with:

    * the position trajectory curving down to the landing pad
    * thrust vectors as arrows along the trajectory
    * the glideslope safety cone shaded behind it
    * a multi-panel time series showing thrust magnitude and altitude

Scenario: a 1500 kg vehicle starts 500 m above and 200 m downrange of the
landing pad, falling at 40 m/s with a 10 m/s horizontal velocity toward
the pad. It must land at the origin with zero velocity in 15 seconds,
respecting engine thrust bounds and a 30 degree glideslope cone.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from starfall.control.convex_guidance import solve_powered_descent


# parameters
MASS = 1500.0
G_VEC = np.array([0.0, -9.81])
T_MIN = 4000.0
T_MAX = 24000.0
T_FINAL = 15.0
N = 75
GLIDESLOPE_DEG = 30.0

r0 = np.array([200.0, 500.0])
v0 = np.array([-10.0, -40.0])


# Solve
print("Solving powered descent problem...")
result = solve_powered_descent(
    r0=r0, v0=v0,
    mass=MASS, g_vec=G_VEC,
    T_min=T_MIN, T_max=T_MAX,
    t_final=T_FINAL, N=N,
    glideslope_angle_deg=GLIDESLOPE_DEG,
)

if result["status"] != "optimal":
    raise RuntimeError(f"Solver did not converge: {result['status']}")

ts = result["ts"]
r = result["r"]
v = result["v"]
T = result["T"]
gamma = result["gamma"]

thrust_mag = np.linalg.norm(T, axis=1)
altitude = r[:, 1]
downrange = r[:, 0]


# Report
print(f"Status:            {result['status']}")
print(f"Final position:    [{r[-1, 0]:.3e}, {r[-1, 1]:.3e}] m")
print(f"Final velocity:    [{v[-1, 0]:.3e}, {v[-1, 1]:.3e}] m/s")
print(f"Peak thrust:       {thrust_mag.max():.2f} N")
print(f"Min thrust:        {thrust_mag.min():.2f} N")
print(f"Fuel proxy:        {gamma.sum() * T_FINAL / N:.2f}")


# Plot
plt.style.use("dark_background")
fig = plt.figure(figsize=(14, 9))

# Main plot: trajectory (large, takes up left half)
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

# Glideslope cone shaded as the safe region
cone_extent = max(np.abs(downrange).max(), altitude.max()) * 1.3
tan_angle = np.tan(np.deg2rad(GLIDESLOPE_DEG))
cone = Polygon(
    [[-cone_extent, cone_extent * tan_angle],
     [-cone_extent, cone_extent * tan_angle * 5],
     [cone_extent, cone_extent * tan_angle * 5],
     [cone_extent, cone_extent * tan_angle]],
    closed=True, facecolor="cyan", alpha=0.05, edgecolor="cyan",
    linestyle="--", linewidth=0.5, label=f"{GLIDESLOPE_DEG:.0f}° glideslope cone",
)
ax1.add_patch(cone)

# Trajectory
ax1.plot(downrange, altitude, color="cyan", linewidth=2.5, label="Trajectory")

arrow_step = 4
arrow_scale = 0.002
for k in range(0, N, arrow_step):
    ax1.arrow(
        downrange[k], altitude[k],
        T[k, 0] * arrow_scale, T[k, 1] * arrow_scale,
        head_width=4, head_length=4, fc="orange", ec="orange",
        alpha=0.85, length_includes_head=True,
    )

# Start and end markers
ax1.scatter([r0[0]], [r0[1]], color="lime", s=120, zorder=5,
            label="Start", edgecolor="white", linewidth=1)
ax1.scatter([0], [0], color="red", marker="X", s=180, zorder=5,
            label="Landing pad", edgecolor="white", linewidth=1)

ax1.set_xlabel("Downrange (m)")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Starfall GNC: Powered Descent via Lossless Convexification")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.25)
ax1.set_aspect("equal")

# Thrust magnitude time series
ax2 = plt.subplot2grid((2, 3), (0, 2))
thrust_times = ts[:-1] + np.diff(ts) / 2
ax2.plot(thrust_times, thrust_mag, color="orange", linewidth=2)
ax2.axhline(T_MIN, color="red", linestyle=":", linewidth=0.8, label="T_min")
ax2.axhline(T_MAX, color="red", linestyle=":", linewidth=0.8, label="T_max")
ax2.fill_between(thrust_times, T_MIN, T_MAX, alpha=0.05, color="red")
ax2.set_ylabel("Thrust magnitude (N)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Bang-bang thrust profile")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.25)

# Altitude time series
ax3 = plt.subplot2grid((2, 3), (1, 2))
ax3.plot(ts, altitude, color="cyan", linewidth=2, label="Altitude")
ax3.plot(ts, downrange, color="magenta", linewidth=2, label="Downrange")
ax3.axhline(0, color="white", linestyle=":", linewidth=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Position (m)")
ax3.set_title("Position vs time")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.25)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "04_powered_descent.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")

plt.show()