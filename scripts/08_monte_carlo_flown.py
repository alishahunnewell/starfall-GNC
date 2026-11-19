"""Monte Carlo of the convex descent guidance, scored on real landing criteria."""

import argparse
import numpy as np
from starfall.control.convex_guidance import solve_powered_descent

NOMINAL_R = np.array([200.0, 500.0])
NOMINAL_V = np.array([-10.0, -40.0])
NOMINAL_MASS = 1500.0
G_VEC = np.array([0.0, -9.81])
T_MIN, T_MAX = 4000.0, 24000.0
T_FINAL, N_STEPS, GLIDESLOPE = 15.0, 75, 30.0
DT = T_FINAL / N_STEPS

SIGMA_POS, SIGMA_VEL, SIGMA_MASS = 20.0, 5.0, 50.0   # true initial dispersion
SIGMA_NAV_POS, SIGMA_NAV_VEL = 5.0, 0.5              # error in the state given to guidance
SIGMA_THRUST = 0.02                                  # fractional thrust magnitude error

PAD_RADIUS = 10.0     # m, allowed horizontal miss at touchdown
MAX_VERT_SPEED = 2.0  # m/s
MAX_HORIZ_SPEED = 1.0 # m/s


def fly(T, mass_true, thrust_scale, r0, v0):
    """Fly the planned thrust open loop with the same discretization as the planner."""
    r, v = r0.copy(), v0.copy()
    for k in range(len(T)):
        r_next = r + DT * v
        v = v + DT * (thrust_scale * T[k] / mass_true + G_VEC)
        r = r_next
        if r[1] <= 0.0:
            return r, v, True
    return r, v, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--no-dispersion", action="store_true",
                    help="sanity check: zero nav/mass/thrust error, should land every feasible trial")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    s = 0.0 if args.no_dispersion else 1.0

    feasible = landed_ok = 0
    reasons = {"no touchdown": 0, "missed pad": 0, "too fast": 0}
    for _ in range(args.trials):
        r0 = NOMINAL_R + rng.normal(0, SIGMA_POS, 2) * s
        v0 = NOMINAL_V + rng.normal(0, SIGMA_VEL, 2) * s
        m_true = NOMINAL_MASS + rng.normal(0, SIGMA_MASS) * s
        eps = rng.normal(0, SIGMA_THRUST) * s
        r_nav = r0 + rng.normal(0, SIGMA_NAV_POS, 2) * s
        v_nav = v0 + rng.normal(0, SIGMA_NAV_VEL, 2) * s

        # guidance only knows the navigated state and the NOMINAL mass
        plan = solve_powered_descent(r0=r_nav, v0=v_nav, mass=NOMINAL_MASS, g_vec=G_VEC,
                                     T_min=T_MIN, T_max=T_MAX, t_final=T_FINAL,
                                     N=N_STEPS, glideslope_angle_deg=GLIDESLOPE)
        if plan["status"] != "optimal":
            continue
        feasible += 1
        r_f, v_f, touched = fly(plan["T"], m_true, 1.0 + eps, r0, v0)
        if not touched:
            reasons["no touchdown"] += 1
        elif abs(r_f[0]) > PAD_RADIUS:
            reasons["missed pad"] += 1
        elif abs(v_f[1]) > MAX_VERT_SPEED or abs(v_f[0]) > MAX_HORIZ_SPEED:
            reasons["too fast"] += 1
        else:
            landed_ok += 1

    n = args.trials
    print(f"Solver feasible:    {feasible}/{n} ({100*feasible/n:.1f}%)")
    print(f"Real landings:      {landed_ok}/{n} ({100*landed_ok/n:.1f}%)")
    print(f"Failures among feasible: {reasons}")


if __name__ == "__main__":
    main()