"""EKF kalman filter altitude estimation with IMU and GPS.

simulated a 1D free fall from 500 m under gravity. The 'truth'
is propagated with exact dynamics. IMU drives the EKF predict
steps. GPS drives the EKF update step. plots overlay truth,
sensor readings, and the EKF estimate to show the filter converging.

"""
from pathlib import Path 

import numpy as np
import matplotlib.pyplot as plt

from starfall.navigation.ekf import ExtendedKalmanFilter
from starfall.navigation.sensors import IMU, GPS

#scenario 
INITIAL_ALTITUDE = 500.0
INITIAL_VELOCITY = 0.0
G = -9.81
T_END = 8.0
DT_IMU = 0.01 #Hz
DT_GPS = 0.10 #Hz
GPS_INTERVAL = int(DT_GPS / DT_IMU) #10 IMU steps per GPS update

#sensors
imu = IMU(noise_std=0.05, bias=0.05, seed=42)
gps = GPS(noise_std=2.0, seed=42)

#EKF dynamics
def f(x, u, dt):
    return np.array([
        x[0] + dt * x[1] + 0.5 * dt**2 * u[0],
        x[1] + dt * u[0],
    ])   
def F(x, u, dt):
    return np.array([[1.0, dt], [0.0, 1.0]])

def h(x):
    return np.array([x[0]])

def H(x):
    return np.array([[1.0, 0.0]])

#starting w a deliberately wrong intial guess to show convergence
x0 = np.array([490.0, 5.0])
P0 = np.diag([100.0, 25.0])
Q = np.diag([0.01, 0.1])
R = np.array([[gps.noise_std**2]])

ekf = ExtendedKalmanFilter(x0, P0, f, F, Q)

#storage 
n_steps = int(T_END / DT_IMU)
ts = np.zeros(n_steps + 1)
truth_alt = np.zeros(n_steps + 1)
truth_vel = np.zeros(n_steps + 1)
est_alt = np.zeros(n_steps + 1)
est_vel = np.zeros(n_steps + 1)
est_alt_std = np.zeros(n_steps + 1)
imu_readings = np.zeros(n_steps)
gps_times = []
gps_readings = []

#initial conditions
truth_alt[0] = INITIAL_ALTITUDE
truth_vel[0] = INITIAL_VELOCITY
est_alt[0] = ekf.x[0]
est_vel[0] = ekf.x[1]
est_alt_std[0] = np.sqrt(ekf.P[0, 0])

#main loop
for k in range(n_steps):
    #advance truth
    truth_alt[k+1] = truth_alt[k] + DT_IMU * truth_vel[k] + 0.5 * DT_IMU**2 * G
    truth_vel[k+1] = truth_vel[k] + DT_IMU * G
    ts[k+1] = ts[k] + DT_IMU

    #IMU reading + predict 
    accel_meas = imu.measure(G)
    imu_readings[k] = accel_meas
    ekf.predict(np.array([accel_meas]), DT_IMU)

    #GPS update (every GPS_INTERVAL steps)
    if (k + 1) % GPS_INTERVAL == 0:
        gps_meas = gps.measure(truth_alt[k+1])
        ekf.update(np.array([gps_meas]), h, H, R)
        gps_times.append(ts[k+1])
        gps_readings.append(gps_meas)

    est_alt[k+1] = ekf.x[0]
    est_vel[k+1] = ekf.x[1]
    est_alt_std[k+1] = np.sqrt(ekf.P[0, 0])

gps_times = np.array(gps_times)
gps_readings = np.array(gps_readings)

#stop at ground impact 
ground_idx =np.where(truth_alt<0)[0]
if len(ground_idx) > 0:
    cutoff = ground_idx[0]
    ts = ts[:cutoff]
    truth_alt = truth_alt[:cutoff]
    truth_vel = truth_vel[:cutoff]
    est_alt = est_alt[:cutoff]
    est_vel = est_vel[:cutoff]
    est_alt_std = est_alt_std[:cutoff]

#report 
alt_err = est_alt - truth_alt  
vel_err = est_vel - truth_vel
print(f"Initial altitude error: {alt_err[0]:+.2f} m")
print(f"Final altitude error:   {alt_err[-1]:+.2f} m")
print(f"Peak altitude error:    {np.max(np.abs(alt_err)):.2f} m")
print(f"RMS altitude error:     {np.sqrt(np.mean(alt_err**2)):.2f} m")
print(f"Initial velocity error: {vel_err[0]:+.2f} m/s")
print(f"Final velocity error:   {vel_err[-1]:+.2f} m/s")

# plot
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

ax1 = axes[0]
ax1.plot(ts, truth_alt, color="lime", linewidth=2, label="Truth", zorder=3)
ax1.scatter(gps_times, gps_readings, color="magenta", s=20, alpha=0.5,
            label="GPS measurements", zorder=2)
ax1.plot(ts, est_alt, color="cyan", linewidth=2, label="EKF estimate", zorder=4)
ax1.fill_between(ts, est_alt - 2 * est_alt_std, est_alt + 2 * est_alt_std,
                 color="cyan", alpha=0.15, label="EKF 2-sigma bound")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Starfall GNC: EKF Altitude Estimation from IMU + GPS")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.25)

ax2 = axes[1]
ax2.plot(ts, truth_vel, color="lime", linewidth=2, label="Truth")
ax2.plot(ts, est_vel, color="cyan", linewidth=2, label="EKF estimate")
ax2.set_ylabel("Vertical velocity (m/s)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.25)

ax3 = axes[2]
ax3.plot(ts, alt_err, color="orange", linewidth=1.5, label="Altitude error")
ax3.plot(ts, vel_err, color="magenta", linewidth=1.5, label="Velocity error")
ax3.axhline(0, color="white", linestyle=":", linewidth=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Estimate error")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.25)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "05_ekf_estimation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")

plt.show()
