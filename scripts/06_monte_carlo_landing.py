"""Monte Carlo dispersion analysis of the convex powered descent guidance.

runs the phase 2c convex landing algorithm N times with randomly
perturbed initial conditions and vehicle mass. for each run, records
the landing position and whether the solver converged. 

result: a landing scatter plot with a 95% confidence ellipse showing 
where the vehicle ends up under aggregate initial condition uncertainty.

"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from starfall.control.convex_guidance import solve_powered_descent

#nominal scenario 
NOMINAL_R = np.array([200.0, 500.0])
NOMINAL_V = np.array([-10.0, -40.0])
NOMINAL_MASS = 1500.0
G_VEC = np.array([0.0, -9.81])
T_MIN = 4000.0
T_MAX = 24000.0
T_FINAL = 15.0
N_STEPS = 75
GLIDESLOPE = 30.0

#monte carlo dispersion parameters (1-sigma)
SIGMA_POS = 20.0 #meters 
SIGMA_VEL = 5.0 #m/s
SIGMA_MASS = 50.0 #kg

N_TRIALS = 1000
SEED = 42

rng = np.random.default_rng(SEED)

#storage for outcomes 
landings = np.zeros((N_TRIALS, 2))
successes = np.zeros(N_TRIALS, dtype=bool)
final_speeds = np.zeros(N_TRIALS)
fuel_costs = np.zeros(N_TRIALS)

print(f"Running {N_TRIALS} Monte Carlo trials...")
for i in range(N_TRIALS):
    #perturbed initial conditions 
    r0 = NOMINAL_R + rng.normal(0.0, SIGMA_POS, size=2)
    v0 = NOMINAL_V + rng.normal(0.0, SIGMA_VEL, size=2)
    mass = NOMINAL_MASS + rng.normal(0.0, SIGMA_MASS)

    #solve
    result = solve_powered_descent(
    r0=r0, v0=v0, mass=mass, g_vec=G_VEC,
    T_min=T_MIN, T_max=T_MAX, t_final=T_FINAL,
    N=N_STEPS, glideslope_angle_deg=GLIDESLOPE
)
    if result["status"] =="optimal":
        successes[i] = True 
        landings[i] = result["r"][-1]
        final_speeds[i] = np.linalg.norm(result["v"][-1])
        fuel_costs[i] = np.sum(result["gamma"]) * T_FINAL / N_STEPS

    if (i +1) %100 == 0:
        rate = 100.0 * np.sum(successes[:i+1]) / (i + 1)
        print(f"  {i+1}/{N_TRIALS} done ({rate:.1f}% converged)")

#stats
n_success = int(np.sum(successes))
success_rate = 100.0 * n_success / N_TRIALS

landed = landings[successes]
mean_landing = landed.mean(axis=0)
cov_landing = np.cov(landed.T)

fuel_mean = fuel_costs[successes].mean()
fuel_std = fuel_costs[successes].std()
speed_max = final_speeds[successes].max()

print()
print(f"Success rate:          {n_success}/{N_TRIALS}  ({success_rate:.1f}%)")
print(f"Mean landing position: [{mean_landing[0]:+.3e}, {mean_landing[1]:+.3e}]")
print(f"Landing covariance:")
print(cov_landing)
print(f"Fuel mean/std:         {fuel_mean:.1f} +/- {fuel_std:.1f}")
print(f"Worst final speed:     {speed_max:.2e} m/s")

#plotting
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

#left panel landing scatter w dispersion ellipse 
ax1 = axes[0]
ax1.scatter(landed[:, 0], landed[:, 1], color="cyan", s=8, alpha=0.4,
            label=f"{n_success} successful landings")
ax1.scatter([0], [0], color="red", marker="X", s=200, zorder=5,
            edgecolor="white", linewidth=1, label="Target")
#95% confidence ellipse from eigen decomposition of covariance matrix 
eigvals, eigvecs = np.linalg.eigh(cov_landing)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

#chi squared value for 95% confidence in 2 degrees of freedom 
chi2_95 = 5.991
width = 2 * np.sqrt(chi2_95 * eigvals[0])
height = 2 * np.sqrt(chi2_95 * eigvals[1])
ellipse = Ellipse(
    xy=mean_landing, width=width, height=height, angle=angle,
    facecolor="none", edgecolor="magenta", linewidth=2,
    linestyle="--", label="95% confidence ellipse",
)

ax1.add_patch(ellipse)
ax1.set_xlabel("Downrange (m)")
ax1.set_ylabel("Altitude (m)")
ax1.set_title(f"Landing dispersion: {N_TRIALS} Monte Carlo trials")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.25)
ax1.set_aspect("equal")

# right panel: fuel distribution histogram
ax2 = axes[1]
ax2.hist(fuel_costs[successes], bins=40, color="orange", alpha=0.7,
         edgecolor="white", linewidth=0.5)
ax2.axvline(fuel_mean, color="cyan", linestyle="--", linewidth=2,
            label=f"Mean = {fuel_mean:.0f}")
ax2.axvline(fuel_mean + fuel_std, color="cyan", linestyle=":", linewidth=1)
ax2.axvline(fuel_mean - fuel_std, color="cyan", linestyle=":", linewidth=1,
            label=f"+/- 1 sigma = {fuel_std:.0f}")
ax2.set_xlabel("Fuel proxy (integrated slack)")
ax2.set_ylabel("Count")
ax2.set_title("Fuel distribution across trials")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.25)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "06_monte_carlo_landing.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")

plt.show()