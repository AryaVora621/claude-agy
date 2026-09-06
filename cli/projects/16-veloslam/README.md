# VeloSLAM: Autonomous Robotics, SLAM & Kinodynamic Motion Planning Engine

[![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib%20only)-success)]()

VeloSLAM is a zero-dependency, portfolio-grade autonomous robotics and mobile navigation engine implemented entirely from first principles in the pure Python standard library. It unifies state estimation, mapping, analytical path synthesis, global kinodynamic search, and local reactive collision avoidance into a cohesive architecture:

* **Extended Kalman Filter SLAM (EKF-SLAM)**: Simultaneous localization of a non-holonomic differential-drive mobile robot and 2D landmark mapping with non-linear kinematic Jacobians, block covariance propagation, and Mahalanobis gating data association.
* **Probabilistic Log-Odds Occupancy Grid Mapping**: Inverse rangefinder LiDAR sensor model with integer Bresenham raycasting line algorithms, recursive Bayesian log-odds accumulation, and inflation safety clearance layers.
* **Analytical Dubins Shortest-Path Solver**: Exact closed-form solver evaluating all 6 canonical word geometries ($LSL$, $RSR$, $LSR$, $RSL$, $RLR$, $LRL$) for non-holonomic minimum-turning-radius vehicles.
* **Kinodynamic Hybrid A* 3D Motion Planner**: Searches continuous $(x, y, \theta)$ vehicle configuration space with discrete steering primitives, 3D discretized duplicate suppression, dual heuristics, and analytical Dubins shots.
* **Dynamic Window Approach (DWA) Local Reactive Planner**: Acceleration-bounded velocity space search $(v, \omega)$, kinetic stopping distance constraints, and multi-objective trajectory rollout evaluation.
* **Unicode Braille Terminal Visualizer**: Sub-pixel 2x4 Braille mapping (`U+2800..U+28FF`) with 24-bit TrueColor ANSI styling rendering occupancy grids, robot poses, heading vectors, covariance ellipses, and vehicle paths.

---

## Architectural Systems & Theoretical Foundations

### 1. Extended Kalman Filter SLAM (EKF-SLAM)
* **Combined State Vector**: Joint estimation of robot pose and $M$ landmark locations:
  $$\mathbf{x}_t = \begin{bmatrix} x_R & y_R & \theta_R & m_{1,x} & m_{1,y} & \dots & m_{M,x} & m_{M,y} \end{bmatrix}^T \in \mathbb{R}^{3 + 2M}$$
* **Kinematic Motion Model & Jacobians**: For forward velocity $v$ and yaw rate $\omega$ over interval $\Delta t$:
  $$x_t = x_{t-1} + v \Delta t \cos(\theta_{t-1}), \quad y_t = y_{t-1} + v \Delta t \sin(\theta_{t-1}), \quad \theta_t = \text{normalize}(\theta_{t-1} + \omega \Delta t)$$
  $$\mathbf{G}_R = \begin{bmatrix} 1 & 0 & -v \Delta t \sin(\theta) \\ 0 & 1 & v \Delta t \cos(\theta) \\ 0 & 0 & 1 \end{bmatrix}$$
* **Block Covariance Update**: Motion propagation updates only robot blocks and cross-covariances, achieving $O(M)$ step complexity rather than $O(M^3)$:
  $$\mathbf{\Sigma}_{RR} \leftarrow \mathbf{G}_R \mathbf{\Sigma}_{RR} \mathbf{G}_R^T + \mathbf{R}, \quad \mathbf{\Sigma}_{RM} \leftarrow \mathbf{G}_R \mathbf{\Sigma}_{RM}$$
* **Range-Bearing Measurement Model**: Sensor observations $z_j = [r, \phi]^T$:
  $$r = \sqrt{\Delta x^2 + \Delta y^2}, \quad \phi = \text{atan2}(\Delta y, \Delta x) - \theta$$
* **Mahalanobis Gating & Data Association**: Evaluates innovation covariance $\mathbf{S}_j = \mathbf{H}_j \mathbf{\Sigma} \mathbf{H}_j^T + \mathbf{Q}$:
  $$D_M^2 = \mathbf{v}_j^T \mathbf{S}_j^{-1} \mathbf{v}_j < \chi_{2, 0.99}^2 = 9.21$$
* **Eigenvalue Covariance Ellipses**: Closed-form 2D characteristic polynomial solver computing semi-major axis, semi-minor axis, and tilt angle for 95% confidence bounds ($s = 2.4477 \sqrt{\lambda}$).

### 2. Probabilistic Log-Odds Occupancy Grid Mapping
* **Bayesian Log-Odds Formulation**:
  $$L(m) = \ln \left( \frac{P(m)}{1 - P(m)} \right), \quad P(m) = 1 - \frac{1}{1 + e^{L(m)}}$$
* **Inverse Sensor Model**:
  * Free space along LiDAR beam: $L_{\text{free}} = \ln(0.35 / 0.65) \approx -0.619$
  * Occupied obstacle endpoint: $L_{\text{occ}} = \ln(0.85 / 0.15) \approx +1.735$
* **Integer Bresenham Raycasting**: Efficient discrete grid traversal without floating-point division along every beam ray.
* **Inflation Safety Layer**: Dilation pass expanding occupied cells by robot radius to ensure guaranteed collision clearance.

### 3. Analytical Dubins Path Solver
* **Canonical Word Geometries**: Evaluates all 6 minimum-curvature curves:
  * Circular-Straight-Circular: $LSL$, $RSR$, $LSR$, $RSL$
  * Circular-Circular-Circular: $RLR$, $LRL$
* **Continuous Trajectory Evaluation**: Parametric sampling $(x(s), y(s), \theta(s))$ yielding exact waypoints for kinematic execution and collision auditing.

### 4. Kinodynamic Hybrid A* 3D Motion Planner
* **Continuous State Space Search**: Explores continuous vehicle poses $(x, y, \theta)$ with discrete steering curvatures $\kappa \in [-\kappa_{\max}, \dots, \kappa_{\max}]$.
* **3D Discretized Duplicate Suppression**: Bins poses into $(X_{\text{grid}}, Y_{\text{grid}}, \Theta_{\text{bin}})$ to eliminate redundant graph expansions while preserving continuous trajectories.
* **Dual Heuristics**: Admissible heuristic taking the maximum of Euclidean distance and obstacle-free Dubins path length:
  $$h(\mathbf{x}) = \max \left( h_{\text{Euclidean}}(\mathbf{x}), h_{\text{Dubins}}(\mathbf{x}) \right)$$
* **Analytic Dubins Shots**: Periodically tests direct minimum-curvature trajectories from candidate search nodes to the exact goal pose, terminating the search early when obstacle-free.

### 5. Dynamic Window Approach (DWA) Local Reactive Navigation
* **Dynamic Velocity Space**: Calculates reachable velocity pairs $(v, \omega)$ considering vehicle acceleration limits:
  $$V_d = [v_c - a \Delta t, v_c + a \Delta t] \cap [v_{\min}, v_{\max}], \quad \Omega_d = [\omega_c - \dot{\omega} \Delta t, \omega_c + \dot{\omega} \Delta t] \cap [-\omega_{\max}, \omega_{\max}]$$
* **Kinetic Stopping Distance Constraint**: Rejects velocities exceeding safe braking threshold:
  $$v \le \sqrt{2 \cdot \text{clearance} \cdot a_{\max}}$$
* **Objective Cost Function**:
  $$G(v, \omega) = \alpha \cdot \text{heading}(v, \omega) + \beta \cdot \text{clearance}(v, \omega) + \gamma \cdot \text{velocity}(v, \omega)$$

---

## Performance Benchmarks

Executed on Apple Silicon (Python 3.13 standard library, single-threaded):

| Subsystem | Metric | Throughput / Latency |
|:---|:---|:---:|
| **Matrix Multiplication** | 20x20 Matrix Dot Product | **2,758 mults/sec** |
| **EKF-SLAM Predict** | Motion update (10 landmarks) | **24,329 predicts/sec** |
| **EKF-SLAM Update** | Observation + Mahalanobis Gating | **568 updates/sec** |
| **Bresenham Raycasting** | 180-beam LiDAR on 200x200 grid | **76,528 rays/sec** |
| **Dubins Path Solver** | Analytical 6-word evaluation | **329,474 paths/sec** |
| **Hybrid A* Planner** | Kinodynamic 3D non-holonomic search | **0.30 ms / plan (3,370 plans/sec)** |
| **DWA Local Planner** | Trajectory rollout & velocity scoring | **67 Hz reactive control loop** |

---

## Project Structure

```
projects/16-veloslam/
├── veloslam/
│   ├── __init__.py           # Unified public API exports
│   ├── linalg.py             # Matrix, Vector, angle normalization, Mahalanobis distance
│   ├── ekf_slam.py           # EKF-SLAM state estimator & covariance propagation
│   ├── occupancy.py          # Log-odds Occupancy Grid & Bresenham raycasting
│   ├── dubins.py             # Analytical Dubins shortest path solver (6 words)
│   ├── hybrid_astar.py       # Kinodynamic 3D Hybrid A* motion planner
│   ├── dwa.py                # Dynamic Window Approach reactive local planner
│   └── visualizer.py         # Sub-pixel Unicode Braille terminal map renderer
├── tests/
│   ├── test_linalg.py        # Vector/matrix operations, inverse, Mahalanobis gating
│   ├── test_ekf_slam.py      # EKF prediction, landmark addition, measurement update
│   ├── test_occupancy.py     # Grid conversion, Bresenham line, log-odds accumulation
│   ├── test_dubins.py        # Straight, U-turn, and 6-word path sampling
│   ├── test_hybrid_astar.py  # Non-holonomic open space & obstacle avoidance planning
│   ├── test_dwa.py           # Dynamic window computation & reactive obstacle avoidance
│   └── test_visualizer.py    # Braille rendering, robot pose, paths, and landmarks
├── benchmarks/
│   └── bench_slam.py         # Complete performance benchmark suite
├── examples/
│   └── robotics_lab.py       # Interactive terminal demonstration with Braille visualization
├── PLAN.md                   # Engineering architecture and mathematical specifications
├── TASK_QUEUE.md             # Autonomous task tracking queue
└── CHECKPOINT_LAST.md        # State persistence and milestone checkpoint
```

---

## Quick Start & Usage

### 1. Run Unit Tests
```bash
PYTHONPATH="." python3 -m unittest discover -s tests
# 27 tests passing in <0.05s
```

### 2. Run Performance Benchmarks
```bash
PYTHONPATH="." python3 benchmarks/bench_slam.py
```

### 3. Run Interactive Demonstration
```bash
PYTHONPATH="." python3 examples/robotics_lab.py
```

### 4. Python API Example
```python
from veloslam.ekf_slam import EKFSLAM
from veloslam.occupancy import OccupancyGrid
from veloslam.hybrid_astar import HybridAStar, Pose2D
from veloslam.dwa import DWAPlanner, RobotState

# 1. Initialize EKF-SLAM
slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)
slam.predict(v=1.0, omega=0.1, dt=0.05)
slam.update([(4.5, 0.2)])  # Range 4.5m, bearing 0.2 rad

# 2. Plan global trajectory with Hybrid A*
grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution=0.2)
planner = HybridAStar(grid, min_turning_radius=1.2)
path = planner.plan(Pose2D(0.0, 0.0, 0.0), Pose2D(8.0, 5.0, 1.57))

# 3. Local reactive avoidance with DWA
dwa = DWAPlanner()
cmd_v, cmd_w, rollout = dwa.plan(RobotState(0.0, 0.0, 0.0), (path[5][0], path[5][1]), grid)
```
