"""Animated visualization of the Phase 2C powered descent trajectory.

Solves the fuel-optimal landing problem, then animates the resulting
trajectory frame by frame using matplotlib's FuncAnimation. Output is
saved as an animated GIF for embedding in the README.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from starfall.control.convex_guidance import solve_powered_descent


# scenario (matches phase 2C)
MASS = 1500.0
G_VEC = np.array([0.0, -9.81])
T_MIN = 4000.0
T_MAX = 24000.0
T_FINAL = 15.0
N = 75
GLIDESLOPE = 30.0

r0 = np.array([200.0, 500.0])
v0 = np.array([-10.0, -40.0])


print("Solving trajectory...")
result = solve_powered_descent(
    r0=r0, v0=v0,
    mass=MASS, g_vec=G_VEC,
    T_min=T_MIN, T_max=T_MAX,
    t_final=T_FINAL, N=N,
    glideslope_angle_deg=GLIDESLOPE,
)

if result["status"] != "optimal":
    raise RuntimeError(f"Solver did not converge: {result['status']}")

positions = result["r"]      # (N+1, 2)
thrusts = result["T"]        # (N, 2)
ts = result["ts"]            # (N+1,)


# animation parameters
FPS = 20
DURATION_SEC = 6.0
N_FRAMES = int(FPS * DURATION_SEC)
FRAME_INDICES = np.linspace(0, len(positions) - 1, N_FRAMES).astype(int)


# figure setup
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(-50, 250)
ax.set_ylim(-30, 550)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)
ax.set_xlabel("Downrange (m)")
ax.set_ylabel("Altitude (m)")
ax.set_title("Starfall GNC: Convex Powered Descent")

# static elements
ax.scatter([0], [0], color="red", marker="X", s=200,
           edgecolor="white", linewidth=1, zorder=5)
ax.text(15, 5, "Landing pad", color="red", fontsize=9)

# dynamic elements (updated each frame)
trail, = ax.plot([], [], color="cyan", linewidth=2, alpha=0.6)
rocket, = ax.plot([], [], marker="^", color="lime", markersize=18,
                  markeredgecolor="white", markeredgewidth=1)
thrust_arrow = ax.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color="orange", lw=2),
)
time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                    color="cyan", fontsize=11, family="monospace",
                    verticalalignment="top")


def init():
    trail.set_data([], [])
    rocket.set_data([], [])
    thrust_arrow.set_position((0, 0))
    thrust_arrow.xy = (0, 0)
    time_text.set_text("")
    return trail, rocket, thrust_arrow, time_text


def animate(frame_num):
    k = FRAME_INDICES[frame_num]

    # trail up to current position
    trail.set_data(positions[:k+1, 0], positions[:k+1, 1])

    # rocket at current position
    x, y = positions[k]
    rocket.set_data([x], [y])

    # thrust arrow (scaled for visibility)
    if k < len(thrusts):
        thrust_scale = 0.006
        tx = thrusts[k, 0] * thrust_scale
        ty = thrusts[k, 1] * thrust_scale
        thrust_arrow.xy = (x + tx, y + ty)
        thrust_arrow.set_position((x, y))
    else:
        thrust_arrow.xy = (x, y)
        thrust_arrow.set_position((x, y))

    # time counter
    time_text.set_text(f"t = {ts[k]:.2f} s\nalt = {y:.1f} m")

    return trail, rocket, thrust_arrow, time_text


print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=N_FRAMES, interval=1000/FPS, blit=False)

out_dir = Path(__file__).parent.parent / "results" / "animations"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "07_landing_animation.gif"

writer = PillowWriter(fps=FPS)
anim.save(out_path, writer=writer, dpi=100)
print(f"Animation saved to: {out_path}")

plt.close(fig)