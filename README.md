# Starfall GNC

> A 6-DOF guidance, navigation, and control simulator for reusable launch vehicles, built from first principles.

![Status](https://img.shields.io/badge/status-Phase%201%3A%20Dynamics-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Goal

Build a complete simulation of a Falcon 9-class booster executing a powered descent and landing, including:

- Full 6-degree-of-freedom rigid body dynamics with quaternion attitude
- Atmospheric and gravitational models
- Sequential controller design: PID → LQR → convex optimization (lossless convexification)
- Extended Kalman Filter for state estimation from realistic IMU/GPS measurements
- Monte Carlo dispersion analysis for terminal landing accuracy

## Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | 6-DOF Rigid Body Dynamics |  In Progress |
| 2 | Control (PID → LQR → Convex Guidance) |  Planned |
| 3 | Navigation (EKF with IMU + GPS) |  Planned |
| 4 | Monte Carlo & Documentation |  Planned |

## About

Built by [Alisha Hunnewell](https://github.com/alishahunnewell), recent graduate at the University of Florida in Astronomy, as part of preparing for graduate study in space systems engineering and a career in launch vehicle GNC.

## Results

### Phase 1: Dynamics Validation

The 6-DOF rigid body dynamics core has been validated end-to-end against
the classical ballistic solution. A 1 kg projectile launched at 45° with
100 m/s in uniform gravity, propagated through the full
`rigid_body_derivative` → RK4 pipeline, matches the closed-form analytical
trajectory to within numerical precision.

![Ballistic validation trajectory](results/plots/01_ballistic_validation.png)

| Quantity | Simulated | Analytical | Error |
|----------|-----------|------------|-------|
| Range (m) | 1018.94 | 1019.37 | 0.042% |
| Apex altitude (m) | 254.84 | 254.84 | 0.000% |
| Flight time (s) | 14.41 | 14.42 | 0.042% |

The remaining error is dominated by impact-time discretization at dt = 10 ms.
Apex altitude matches analytically to the printed precision.

### Phase 2A: PID Altitude Hold

A PID controller drives a 1 kg vehicle from 80 m initial altitude up to a
100 m hover setpoint, with thrust saturation at 4× weight. The controller
saturates against the upper thrust limit during the initial climb,
overshoots ~29 m, then settles to within 1.2 cm of the setpoint under
conditional anti-windup.

![PID altitude hold step response](results/plots/02_pid_hover.png)

| Quantity | Value |
|----------|-------|
| Steady-state altitude error | 0.012 m |
| Steady-state velocity error | 0.002 m/s |
| Peak overshoot | 29.1 m |
| Settling time (to ±0.5 m) | ~10 s |
| Gains | Kp=6, Ki=1, Kd=4.5 |

### Phase 2B: LQR Optimal Control

LQR (Linear Quadratic Regulator) optimal feedback gains are derived by
solving the continuous algebraic Riccati equation around the hover
equilibrium. The same 80 m → 100 m hover problem is rerun, this time
using LQR-derived gains instead of hand-tuned PID gains. The LQR
controller outperforms PID across every metric.

![PID vs LQR step response](results/plots/03_lqr_vs_pid_hover.png)

| Quantity | PID | LQR |
|----------|-----|-----|
| Steady-state altitude error | 0.012 m | 0.000 m |
| Peak overshoot | 29.1 m | 11.3 m |
| Max abs error | 29.1 m | 20.0 m |
| Final velocity | 0.002 m/s | 0.000 m/s |

LQR achieves ~2.6× less overshoot, no steady-state error, and a smoother
control signal, all derived from a quadratic cost J = ∫(x'Qx + u'Ru)dt
with Q = diag(10, 1), R = 0.1 against the linearized hover dynamics.
### Phase 2C: Convex Powered Descent Guidance

The fuel-optimal landing problem is formulated as a convex optimization
using lossless convexification (Açıkmeşe & Ploen 2007). The non-convex
thrust magnitude constraint is replaced with a slack-variable formulation
that the theorem guarantees is exact at optimality. CVXPY discretizes the
problem into a 354-variable second-order cone program and solves it in
under 100 ms.

A 1500 kg vehicle starting 500 m above and 200 m downrange of the landing
pad, with 40 m/s vertical and 10 m/s horizontal velocity, lands at the
origin in 15 seconds while respecting a 30° glideslope cone and engine
thrust bounds (4 kN minimum, 24 kN maximum).

![Powered descent trajectory](results/plots/04_powered_descent.png)

| Quantity | Value |
|----------|-------|
| Solver status | optimal |
| Final position error | < 10⁻¹¹ m |
| Final velocity error | < 10⁻¹⁰ m/s |
| Peak thrust | 24,000 N (T_max saturation) |
| Min thrust | 4,000 N (T_min, slack active) |
| Bang-bang structure | max → min → max, classic fuel-optimal |

The bang-bang thrust profile and the slack-bound activation at T_min are
the visible signatures of lossless convexification at work. This is the
same algorithm family used by SpaceX for Falcon 9 first-stage landing
burns (Blackmore 2013).
### Phase 3: Extended Kalman Filter Navigation

An Extended Kalman Filter fuses high-rate noisy IMU acceleration
measurements (100 Hz) with slower GPS position measurements (10 Hz)
to estimate altitude and vertical velocity in real time. The filter
starts with a deliberately wrong initial guess (10 m altitude offset,
5 m/s velocity offset) to demonstrate convergence.

![EKF altitude estimation](results/plots/05_ekf_estimation.png)

| Quantity | Value |
|----------|-------|
| Initial altitude error | -10.00 m (deliberate) |
| Final altitude error | +0.15 m |
| RMS altitude error | 1.28 m |
| Initial velocity error | +5.00 m/s (deliberate) |
| Final velocity error | +0.49 m/s |

The EKF's 2-sigma uncertainty band shrinks rapidly after the first
GPS update, showing the filter automatically re-weighting measurement
information against dynamics propagation. This is the same predict-
update algorithm family used in the Apollo digital autopilot, ISS
proximity operations, and every modern spacecraft that fuses sensors.
### Phase 4: Monte Carlo Dispersion Analysis

The Phase 2C convex guidance is subjected to 1000 Monte Carlo trials
with randomized initial position (±20 m, 1-sigma), velocity (±5 m/s,
1-sigma), and vehicle mass (±50 kg, 1-sigma). Each trial runs the full
convex optimization pipeline independently. This is the standard
deliverable used for mission approval in real aerospace GNC.

![Monte Carlo landing dispersion](results/plots/06_monte_carlo_landing.png)

| Quantity | Value |
|----------|-------|
| Successful convergence rate | 99.5% (995 / 1000) |
| Mean landing position | ~8 × 10⁻¹² m (picometer bias) |
| 95% dispersion ellipse | sub-nanometer axes |
| Fuel mean ± std | 284,430 ± 12,254 (4.3% CoV) |
| Worst terminal velocity | 4 × 10⁻⁸ m/s |

The algorithm handles the full ±3-sigma uncertainty envelope with
essentially no degradation in terminal accuracy. The 0.5% failure
mode corresponds to the most extreme velocity perturbations
(3-sigma tails), which is the expected behavior for a well-designed
guidance system.
## References

- Wie, *Space Vehicle Dynamics and Control*
- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*
- Açıkmeşe & Ploen (2007), "Convex Programming Approach to Powered Descent Guidance for Mars Landing"
- Blackmore (2013), "Lossless Convexification of Optimal Control Problems"
