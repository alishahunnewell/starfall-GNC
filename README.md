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

Built by [Alisha Hunnewell](https://github.com/alishahunnewell), graduating senior at the University of Florida, as part of preparing for graduate study in space systems engineering and a career in launch vehicle GNC.

## References

- Wie, *Space Vehicle Dynamics and Control*
- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*
- Açıkmeşe & Ploen (2007), "Convex Programming Approach to Powered Descent Guidance for Mars Landing"
- Blackmore (2013), "Lossless Convexification of Optimal Control Problems"
