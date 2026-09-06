# Project 25 Architecture Plan: Avionix

## Autonomous Quadrotor Flight Dynamics, SE(3) Geometric Tracking Control, Minimum-Snap Trajectory Generation, and Primary Flight Display Avionics

### Executive Summary
Avionix is a full-stack aerial robotics, flight simulation, and non-linear control framework built entirely in the pure Python standard library. It models full 6-DOF rigid-body aerial dynamics on the Special Euclidean Group $SE(3)$, provides differential-flatness-based minimum-snap polynomial trajectory optimization, implements the benchmark Lee-Leok-McClamroch geometric tracking controller on the Lie group $SO(3)$, provides a multi-rate Extended Kalman Filter (EKF) sensor fusion engine, and renders an interactive Primary Flight Display (PFD) and 3D sub-pixel Unicode Braille trajectory visualizer in the terminal.

---

### Module Architecture

1. **6-DOF Quadrotor Dynamics & Kinematics (`avionix/dynamics.py`)**:
   - Rigid-body Newton-Euler equations of motion in world frame $\mathcal{W}$ and body frame $\mathcal{B}$:
     $$\dot{\mathbf{p}} = \mathbf{v}$$
     $$m \dot{\mathbf{v}} = -m g \mathbf{e}_3 + \mathbf{R} \mathbf{f}_b + \mathbf{f}_{\text{ext}}$$
     $$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$$
     $$\mathbf{J} \dot{\boldsymbol{\omega}} = \boldsymbol{\tau}_b - \boldsymbol{\omega} \times (\mathbf{J} \boldsymbol{\omega}) - \sum_{i=1}^4 \mathbf{J}_{r} (\boldsymbol{\omega} \times \mathbf{e}_3) (-1)^i \Omega_i$$
   - Quaternion algebra with singularity-free integration, normalisation, and conversion to rotation matrix $\mathbf{R} \in SO(3)$.
   - Motor rotor dynamics: First-order motor lag $\tau_m \dot{\Omega}_i = \Omega_{i,\text{cmd}} - \Omega_i$, quadratic thrust $f_i = c_T \Omega_i^2$, and aerodynamic drag torque $\tau_{d,i} = c_Q \Omega_i^2$.
   - Aerodynamic drag: Translational parasitic drag $\mathbf{f}_d = -\frac{1}{2} \rho C_d A \|\mathbf{v}\| \mathbf{v}$ and rotor blade-flapping induced drag.
   - Integrators: Symplectic/semi-implicit Euler and 4th-Order Runge-Kutta (RK4) for high-fidelity numerical integration.

2. **Minimum-Snap Trajectory Generation & Differential Flatness (`avionix/trajectory.py`)**:
   - Differential Flatness of Quadrotors: The 4 flat outputs $\boldsymbol{\sigma}(t) = [x(t), y(t), z(t), \psi(t)]^T$ (3D position and yaw angle) uniquely determine all state variables $(\mathbf{p}, \mathbf{v}, \mathbf{R}, \boldsymbol{\omega})$ and control inputs $(f, \boldsymbol{\tau})$ without integrating differential equations:
     $$\mathbf{z}_b = \frac{\mathbf{t}}{\|\mathbf{t}\|}, \quad \text{where } \mathbf{t} = m(\ddot{\mathbf{p}} + g \mathbf{e}_3)$$
     $$\mathbf{y}_b = \frac{\mathbf{z}_b \times \mathbf{x}_c}{\|\mathbf{z}_b \times \mathbf{x}_c\|}, \quad \mathbf{x}_c = [\cos\psi, \sin\psi, 0]^T$$
     $$\mathbf{x}_b = \mathbf{y}_b \times \mathbf{z}_b$$
     $$\mathbf{R} = [\mathbf{x}_b, \mathbf{y}_b, \mathbf{z}_b]$$
     $$f = \mathbf{t} \cdot \mathbf{z}_b$$
     Angular velocity $\boldsymbol{\omega}$ is recovered from $\mathbf{h}_\omega = \frac{m}{f} (\mathbf{p}^{(3)} - (\mathbf{z}_b \cdot \mathbf{p}^{(3)}) \mathbf{z}_b)$.
   - Minimum-Snap Polynomial Spline Optimization: Piecewise 7th-degree (order 8) polynomials minimizing snap (4th derivative of position $\int \|\mathbf{p}^{(4)}(t)\|^2 dt$):
     $$p_k(t) = \sum_{j=0}^7 c_{k,j} t^j$$
     Continuous position ($C^0$), velocity ($C^1$), acceleration ($C^2$), jerk ($C^3$), and snap ($C^4$) across multi-waypoint segments.
     Solved via linear system of continuity and boundary equations without external QP libraries.

3. **SE(3) Geometric Tracking Control & Motor Mixer (`avionix/control.py`)**:
   - Lee, Leok, McClamroch (2010) Geometric Tracking Control on $SE(3)$:
     - Position Tracking Error: $\mathbf{e}_x = \mathbf{x} - \mathbf{x}_d, \quad \mathbf{e}_v = \mathbf{v} - \mathbf{v}_d$
     - Desired Thrust Vector: $\mathbf{A} = -k_x \mathbf{e}_x - k_v \mathbf{e}_v - m g \mathbf{e}_3 + m \ddot{\mathbf{x}}_d$
     - Total Collective Thrust: $f = -(\mathbf{A} \cdot \mathbf{R} \mathbf{e}_3)$
     - Desired Orientation: $\mathbf{R}_d = [\mathbf{b}_{1d}, \mathbf{b}_{2d}, \mathbf{b}_{3d}]$ computed from normalized $\mathbf{A}$ and yaw angle $\psi_d$.
     - Attitude Error Vector on $SO(3)$:
       $$\mathbf{e}_R = \frac{1}{2} (\mathbf{R}_d^T \mathbf{R} - \mathbf{R}^T \mathbf{R}_d)^\vee$$
       where $(\cdot)^\vee$ is the un-skew symmetric map from $\mathfrak{so}(3)$ to $\mathbb{R}^3$.
     - Angular Rate Error: $\mathbf{e}_\Omega = \boldsymbol{\omega} - \mathbf{R}^T \mathbf{R}_d \boldsymbol{\omega}_d$
     - Control Moments:
       $$\boldsymbol{\tau} = -k_R \mathbf{e}_R - k_\Omega \mathbf{e}_\Omega + \boldsymbol{\omega} \times (\mathbf{J} \boldsymbol{\omega}) - \mathbf{J} (\hat{\boldsymbol{\omega}} \mathbf{R}^T \mathbf{R}_d \boldsymbol{\omega}_d - \mathbf{R}^T \mathbf{R}_d \dot{\boldsymbol{\omega}}_d)$$
   - Cascaded Nonlinear PID Controller: Alternative dual-loop architecture (Outer-loop position P/PD, Inner-loop attitude quaternion/Euler PD, Rate feedforward).
   - Motor Mixer Matrix ($X$-frame quadrotor configuration):
     $$\begin{bmatrix} f \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} = \begin{bmatrix} c_T & c_T & c_T & c_T \\ -d c_T & -d c_T & d c_T & d c_T \\ -d c_T & d c_T & d c_T & -d c_T \\ -c_Q & c_Q & -c_Q & c_Q \end{bmatrix} \begin{bmatrix} \Omega_1^2 \\ \Omega_2^2 \\ \Omega_3^2 \\ \Omega_4^2 \end{bmatrix}$$
     Inverted to map desired wrench $[f, \boldsymbol{\tau}]^T$ to per-motor commands with anti-windup clamping.

4. **Sensor Simulation & Multi-Rate EKF State Estimator (`avionix/estimation.py`)**:
   - Simulated Sensors:
     - 3-axis MEMS accelerometer with gravity, centrifugal acceleration, and bias noise: $\mathbf{a}_m = \mathbf{R}^T (\dot{\mathbf{v}} + g \mathbf{e}_3) + \mathbf{b}_a + \boldsymbol{\eta}_a$.
     - 3-axis MEMS rate gyroscope with random walk bias drift: $\boldsymbol{\omega}_m = \boldsymbol{\omega} + \mathbf{b}_g + \boldsymbol{\eta}_g, \quad \dot{\mathbf{b}}_g = \boldsymbol{\eta}_{bg}$.
     - Barometric altimeter: $h_m = -p_z + \eta_h$.
     - 3-axis magnetometer: $\mathbf{m}_m = \mathbf{R}^T \mathbf{m}_{\text{earth}} + \boldsymbol{\eta}_m$.
     - GPS position and velocity receiver (low rate, 10 Hz): $\mathbf{p}_{\text{gps}} = \mathbf{p} + \boldsymbol{\eta}_p, \quad \mathbf{v}_{\text{gps}} = \mathbf{v} + \boldsymbol{\eta}_v$.
   - Multi-Rate Extended Kalman Filter (EKF):
     - 15-state error formulation: $\mathbf{x} = [\mathbf{p}^T, \mathbf{v}^T, \mathbf{q}^T, \mathbf{b}_a^T, \mathbf{b}_g^T]^T$.
     - Error-state attitude representation (ES-EKF) avoiding quaternion covariance degeneracy.
     - Propagation at IMU rate (200-500 Hz).
     - Asynchronous measurement updates for Barometer, Magnetometer, and GPS.

5. **Primary Flight Display (PFD) & 3D Sub-Pixel Braille Visualizer (`avionix/avionics_pfd.py`)**:
   - Electronic Flight Instrument System (EFIS) Primary Flight Display:
     - Artificial Horizon: Pitch ladder ticks ($+10^\circ, +20^\circ, -10^\circ, -20^\circ$), roll pointer scale with bank angle arc, and central aircraft boresight reticle.
     - Airspeed Tape: Left vertical tape showing calibrated airspeed (CAS), trend vector, and target speed bug.
     - Altimeter Tape: Right vertical tape showing barometric altitude, vertical speed indicator (VSI) needle, and altitude bug.
     - Heading Compass Rose: Bottom directional heading tape ($000^\circ$ to $359^\circ$) with waypoint tracking bug.
     - Flight Mode Annunciator (FMA): Top status bar showing autopilot mode (NAV, ALT, ATT, RTL, LAND) and battery/motor health.
   - 3D Sub-Pixel Unicode Braille Trajectory Visualizer:
     - 3D wireframe waypoints and actual flight path rasterized onto a $2 \times 4$ Braille canvas (`U+2800..U+28FF`).
     - Real-time quadrotor attitude gimbal representation with arm orientation indicators.

6. **Flight Autopilot & Mission Executive (`avionix/autopilot.py`)**:
   - State machine: `DISARMED`, `ARMED`, `TAKEOFF`, `WAYPOINT_NAV`, `TRAJECTORY_TRACKING`, `ACRO_FLIP`, `RETURN_TO_HOME`, `EMERGENCY_LAND`.
   - Disturbance Observer (DOB): Estimates external wind gust forces $\hat{\mathbf{f}}_{\text{wind}}$ via momentum residual filtering and actively trims feedforward thrust.
   - Pre-configured aerobatic flight missions:
     - Aggressive High-Speed Slalom.
     - 3D Lemniscate (Figure-8) Trajectory.
     - Barrel Roll / 360-degree Acrobatic Flip.
     - Station-keeping hovering in violent turbulent wind gusts.

---

### Verification and Quality Standards
- Comprehensive unit tests covering dynamics conservation, quaternion algebra, minimum-snap optimization continuity, geometric control stability, EKF convergence, and visualizer rendering.
- Zero external dependencies: pure Python standard library exclusively.
- Strict avoidance of em dashes across all code, docstrings, outputs, and documentation.
- High-throughput performance microbenchmarks (>100k dynamics steps/s, >10k EKF steps/s).
- Full showcase integration into `showcase.py`, `projects.md`, and `tracker/data.json`.
