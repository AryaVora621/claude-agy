# VeloSLAM: Architectural Design Specification
## Autonomous Robotics, Simultaneous Localization and Mapping (SLAM) & Kinodynamic Motion Planning Engine

**Domain**: Autonomous Mobile Robotics / State Estimation / Computational Geometry / Path Planning  
**Target Platform**: Python 3.10+ (Standard Library Only - Zero External Dependencies)  
**Status**: Architecture & Specification  

---

## 1. Executive Summary

VeloSLAM is a comprehensive, production-grade autonomous robotics framework designed and implemented entirely from first principles in pure Python. It implements the foundational software stack of an autonomous ground vehicle (AGV):
1. **Linear Algebra & Covariance Matrix Engine (`veloslam/linalg.py`)**: Vector and matrix operations, matrix inversion, determinants, eigen/covariance ellipse parameters, and Mahalanobis gating.
2. **Extended Kalman Filter SLAM (`veloslam/ekf_slam.py`)**: Simultaneous estimation of unicycle robot pose $(x, y, \theta)$ and unknown 2D landmark locations with non-linear motion and measurement Jacobians.
3. **Probabilistic Occupancy Grid Mapping (`veloslam/occupancy.py`)**: Log-odds Bayesian mapping using Bresenham LiDAR beam raycasting and inverse rangefinder sensor models.
4. **Kinodynamic Hybrid A* & Dubins Curves (`veloslam/hybrid_astar.py`)**: Continuous non-holonomic vehicle path planning with minimum turning radius constraints and analytical Dubins curve primitives (LSL, RSR, LSR, RSL, RLR, LRL).
5. **Dynamic Window Approach (DWA) Local Planner (`veloslam/dwa.py`)**: Real-time velocity space $[v, \omega]$ trajectory rollout and dynamic collision avoidance.
6. **High-Resolution Unicode Braille Visualizer (`veloslam/visualizer.py`)**: 2x4 sub-pixel Braille terminal renderer displaying maps, LiDAR scans, vehicle trajectories, and covariance uncertainty ellipses in 24-bit TrueColor ANSI.

---

## 2. Mathematical Formulations

### 2.1 Extended Kalman Filter SLAM (EKF-SLAM)

#### State Representation
The joint state vector $\mathbf{x}_t \in \mathbb{R}^{3 + 2M}$ tracks the robot pose and $M$ 2D point landmarks:
$$\mathbf{x}_t = \begin{bmatrix} x_R \\ y_R \\ \theta_R \\ m_{1,x} \\ m_{1,y} \\ \vdots \\ m_{M,x} \\ m_{M,y} \end{bmatrix}, \quad \mathbf{\Sigma}_t = \begin{bmatrix} \mathbf{\Sigma}_{RR} & \mathbf{\Sigma}_{RM} \\ \mathbf{\Sigma}_{MR} & \mathbf{\Sigma}_{MM} \end{bmatrix} \in \mathbb{R}^{(3+2M) \times (3+2M)}$$

#### Kinematic Motion Model
Given control input $\mathbf{u}_t = [v_t, \omega_t]^T$ (linear and angular velocity) over time step $\Delta t$:
$$\mathbf{x}_{R, t} = g(\mathbf{x}_{R, t-1}, \mathbf{u}_t) = \begin{bmatrix} x_{t-1} + v_t \Delta t \cos(\theta_{t-1}) \\ y_{t-1} + v_t \Delta t \sin(\theta_{t-1}) \\ \text{normalize\_angle}(\theta_{t-1} + \omega_t \Delta t) \end{bmatrix}$$

#### Motion Jacobian
$$G_t = \frac{\partial g}{\partial \mathbf{x}} = \begin{bmatrix} 1 & 0 & -v_t \Delta t \sin(\theta_{t-1}) & \mathbf{0}_{3 \times 2M} \\ 0 & 1 & v_t \Delta t \cos(\theta_{t-1}) & \mathbf{0}_{3 \times 2M} \\ 0 & 0 & 1 & \mathbf{0}_{3 \times 2M} \\ \mathbf{0}_{2M \times 3} & \mathbf{0}_{2M \times 3} & \mathbf{0}_{2M \times 3} & \mathbf{I}_{2M \times 2M} \end{bmatrix}$$

#### Measurement Model (Range-Bearing LiDAR)
For an observed landmark $j$:
$$\mathbf{z}_{t}^j = h(\mathbf{x}_t, j) = \begin{bmatrix} r_j \\ \phi_j \end{bmatrix} = \begin{bmatrix} \sqrt{(m_{j,x} - x_R)^2 + (m_{j,y} - y_R)^2} \\ \text{atan2}(m_{j,y} - y_R, m_{j,x} - x_R) - \theta_R \end{bmatrix}$$

#### Measurement Jacobian $H_t^j$
$$\Delta x = m_{j,x} - x_R, \quad \Delta y = m_{j,y} - y_R, \quad q = \Delta x^2 + \Delta y^2, \quad r = \sqrt{q}$$
$$H_t^j = \frac{1}{q} \begin{bmatrix} -r \Delta x & -r \Delta y & 0 & \dots & r \Delta x & r \Delta y & \dots \\ \Delta y & -\Delta x & -q & \dots & -\Delta y & \Delta x & \dots \end{bmatrix}$$

#### Data Association via Mahalanobis Distance
For each candidate landmark $j$:
$$d_M^2 = (\mathbf{z} - \hat{\mathbf{z}}_j)^T \mathbf{S}_j^{-1} (\mathbf{z} - \hat{\mathbf{z}}_j), \quad \mathbf{S}_j = H_j \mathbf{\Sigma} H_j^T + Q$$
If $\min_j d_M^2 > \gamma_{\text{gate}}$ (chi-square threshold), a new landmark is initialized.

---

### 2.2 Probabilistic Occupancy Grid Mapping

* **Log-Odds Update Formula**:
  $$L_t(m_i) = L_{t-1}(m_i) + \text{inv\_sensor}(\mathbf{x}_t, z_t) - L_0$$
* **Probability Recovery**:
  $$P(m_i = 1) = 1 - \frac{1}{1 + \exp(L(m_i))}$$
* **Inverse Sensor Model**:
  * Free space along ray: $L_{\text{free}} = \ln \frac{0.35}{0.65} \approx -0.619$
  * Occupied space at endpoint: $L_{\text{occ}} = \ln \frac{0.85}{0.15} \approx +1.735$
* **Bresenham Raycasting**: Efficient discrete grid traversal connecting sensor origin $(c_x, c_y)$ to endpoint $(e_x, e_y)$.

---

### 2.3 Kinodynamic Hybrid A* & Dubins Curves

* **Continuous State Representation**: Nodes have floating-point coordinates $(x, y, \theta)$, but are indexed in a 3D discretized hash table $(X_{\text{idx}}, Y_{\text{idx}}, \Theta_{\text{idx}})$ for closed-set duplicate suppression.
* **Analytical Dubins Path**: Connects two directed poses with minimum curvature radius $\rho_{\min}$ across 6 word types:
  1. $LSL$: Left-Straight-Left
  2. $RSR$: Right-Straight-Right
  3. $LSR$: Left-Straight-Right
  4. $RSL$: Right-Straight-Left
  5. $RLR$: Right-Left-Right
  6. $LRL$: Left-Right-Left
* **Analytic Shot**: At every $K$ node expansions, computes exact Dubins trajectory to the goal. If collision-free against the occupancy grid, terminates search instantly.
* **Dual Heuristic**: $h(n) = \max(h_{\text{2D\_Euclidean}}(n), h_{\text{Dubins}}(n))$.

---

### 2.4 Dynamic Window Approach (DWA) Local Planning

* **Velocity Search Space**:
  $$V_d = [v_c - a_{\max} \Delta t, v_c + a_{\max} \Delta t] \cap [v_{\min}, v_{\max}]$$
  $$\Omega_d = [\omega_c - \alpha_{\max} \Delta t, \omega_c + \alpha_{\max} \Delta t] \cap [-\omega_{\max}, \omega_{\max}]$$
* **Objective Cost Function**:
  $$G(v, \omega) = w_1 \cdot \text{heading}(v, \omega) + w_2 \cdot \text{dist\_to\_obs}(v, \omega) + w_3 \cdot v$$
* **Kinetic Stopping Constraint**:
  $$v \le \sqrt{2 \cdot \text{dist}_{\text{obs}} \cdot a_{\max}}$$

---

## 3. Directory Structure

```
projects/16-veloslam/
├── veloslam/
│   ├── __init__.py           # Package exports
│   ├── linalg.py             # Pure Python matrix/vector engine & Mahalanobis distance
│   ├── ekf_slam.py           # Extended Kalman Filter SLAM & landmark management
│   ├── occupancy.py          # Log-odds Bayesian occupancy grid & Bresenham raycaster
│   ├── dubins.py             # Exact minimum-length Dubins curve solver (LSL/RSR/LSR/RSL/RLR/LRL)
│   ├── hybrid_astar.py       # Kinodynamic 3D continuous Hybrid A* motion planner
│   ├── dwa.py                # Dynamic Window Approach velocity-space local planner
│   └── visualizer.py         # Sub-pixel Unicode Braille grid & trajectory visualizer
├── tests/
│   ├── __init__.py
│   ├── test_linalg.py        # Matrix invert, determinant, Mahalanobis, angles
│   ├── test_ekf_slam.py      # Motion prediction, measurement updates, loop closure
│   ├── test_occupancy.py     # Log-odds updates, raycasting, probability thresholds
│   ├── test_dubins.py        # All 6 Dubins words, curvature radius, length optimality
│   ├── test_hybrid_astar.py  # Obstacle avoidance, analytic shot, non-holonomic path
│   └── test_dwa.py           # Velocity sampling, collision avoidance, goal attraction
├── benchmarks/
│   └── bench_slam.py         # Performance benchmarks (EKF, Raycast, Dubins, Hybrid A*)
├── examples/
│   └── robotics_lab.py       # Full interactive simulation & Braille visualization
├── PLAN.md                   # This document
├── CHECKPOINT_LAST.md        # State tracking
└── README.md                 # Complete documentation
```

---

## 4. Implementation Phases

1. **Phase 1: Linear Algebra & Matrix Math (`veloslam/linalg.py`)**:
   Vectors, matrices, transpose, multiplication, determinant, Gauss-Jordan inverse, 2D rotation matrices, angle normalization, and Mahalanobis distance.
2. **Phase 2: EKF-SLAM Engine (`veloslam/ekf_slam.py`)**:
   State vector $[x, y, \theta, m_1, \dots, m_k]$, motion prediction, observation Jacobians, Kalman gain, covariance update, and Mahalanobis gating.
3. **Phase 3: Occupancy Grid & LiDAR Raycasting (`veloslam/occupancy.py`)**:
   Log-odds grid, inverse sensor model, Bresenham 2D line traversal, and obstacle extraction.
4. **Phase 4: Dubins Curves & Kinodynamic Hybrid A* (`veloslam/dubins.py`, `veloslam/hybrid_astar.py`)**:
   Analytical Dubins 6-path solver, continuous vehicle state search, collision checking, and analytic shot.
5. **Phase 5: DWA Local Planner & Braille Visualizer (`veloslam/dwa.py`, `veloslam/visualizer.py`)**:
   Velocity window rollout, obstacle distance scoring, and sub-pixel Braille terminal map renderer.
6. **Phase 6: Verification, Testing & Benchmarking (`tests/`, `benchmarks/`, `examples/`)**:
   Full test suite, performance benchmarks, and interactive simulation.
