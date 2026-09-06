# NovaPhysics: 2D Rigid Body Dynamics, GJK/EPA Collision Engine & Constraint Solver

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Stdlib)-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-25%20Passing-brightgreen.svg)]()

NovaPhysics is a production-grade 2D physics simulation engine, convex collision detection system, and mechanical constraint solver written from first principles in the pure Python standard library with zero third-party dependencies.

It implements the mathematical foundation of modern physics engines (such as Box2D and PhysX): Symplectic Euler integration, Gilbert-Johnson-Keerthi (GJK) convex distance queries, Expanding Polytope Algorithm (EPA) contact manifolds, Projected Gauss-Seidel sequential impulse resolution with warm starting, Dynamic AABB tree broadphase pruning, Position-Based Dynamics (PBD) Verlet cloth, and a sub-pixel Unicode Braille terminal rasterizer.

---

## Key Features

- **Symplectic (Semi-Implicit) Euler Integrator**: Updates velocity before position to preserve Hamiltonian phase space volume and prevent spurious energy accumulation in orbital and oscillatory systems.
- **Convex Geometry & Support Mappings**: Supports circles, oriented bounding boxes (OBBs), and arbitrary counter-clockwise convex polygons with exact center of mass and moment of inertia computed via triangular decomposition.
- **Dynamic AABB Bounding Volume Hierarchy**: Implements an incremental surface-area heuristic (SAH) dynamic bounding volume tree with fattened bounding boxes, reducing $O(N^2)$ pairwise collision overhead to $O(N \log N)$.
- **GJK (Gilbert-Johnson-Keerthi) Collision Algorithm**: Evaluates convex shape intersection without computing explicit polygon intersections by evolving a 2D simplex enclosing the origin in Minkowski difference space $A \ominus B$.
- **EPA (Expanding Polytope Algorithm)**: Iteratively expands the Minkowski simplex along closest edge normals to determine exact penetration depth and contact normal pointing from body A to body B.
- **Feature Edge Clipping**: Performs Sutherland-Hodgman contact feature clipping between reference and incident edges to generate stable 2-point contact manifolds for box-on-box stacking.
- **Sequential Impulse Solver & Warm Starting**: Solves normal non-penetration impulses, restitution bounce, and Coulomb friction cones ($|J_t| \le \mu J_n$). Employs Baumgarte overlap stabilization and warm starting across time steps to prevent jittering or sinking stacks.
- **Mechanical Joint Constraints**:
  - `DistanceJoint`: Conserves exact metric distance between two anchor points (pendulums, linkages).
  - `RevoluteJoint`: Constrains two bodies to a shared pivot point with free angular rotation (hinges, ragdolls).
  - `SpringJoint`: Harmonic oscillator with tunable stiffness and viscous damping.
- **Position-Based Dynamics (PBD) Verlet Cloth**: Simulates tearable 2D cloth meshes with structural, shear, and bending springs, interactive cutting, wind turbulence, and circular obstacle repulsion.
- **Sub-Pixel Unicode Braille Terminal Visualizer**: 2x4 sub-pixel rasterization ($152 \times 96$ effective resolution) using Unicode Braille patterns (`U+2800` - `U+28FF`) with live kinetic energy tracking, bodies, and joint telemetry.

---

## Architectural Layout

```
projects/12-novaphysics/
|-- novaphysics/
|   |-- __init__.py          # Public package exports
|   |-- math2d.py            # Vec2, Mat22, Transform2D, rotation utilities
|   |-- shapes.py            # Shape, Circle, Polygon, Box, AABB, moments of inertia
|   |-- body.py              # RigidBody, BodyType (Static, Dynamic, Kinematic), Material
|   |-- broadphase.py        # Dynamic AABB Tree with SAH insertion and pair queries
|   |-- gjk.py               # Gilbert-Johnson-Keerthi convex intersection algorithm
|   |-- epa.py               # Expanding Polytope Algorithm for MTV and contact normals
|   |-- contact.py           # ContactManifold, ContactPoint, and edge clipping
|   |-- solver.py            # SequentialImpulseSolver, Baumgarte stabilization, warm starting
|   |-- joints.py            # DistanceJoint, RevoluteJoint, SpringJoint
|   |-- world.py             # World simulation manager, sub-stepping, symplectic stepping
|   |-- cloth.py             # VerletParticle, DistanceConstraint, ClothMesh
|   `-- visualizer.py        # BrailleCanvas (2x4 sub-pixels) and PhysicsRenderer
|-- tests/
|   |-- test_math2d.py       # 2D vector and matrix tests
|   |-- test_shapes.py       # Centroid, area, and moment of inertia validation
|   |-- test_broadphase.py   # Dynamic tree insertion, rebalancing, and query tests
|   |-- test_gjk_epa.py      # GJK intersection and EPA penetration tests
|   |-- test_solver.py       # Ground contact, momentum conservation, stacking stability
|   |-- test_joints.py       # Distance, revolute, and spring constraint tests
|   `-- test_cloth.py        # PBD particle integration, tearing, and obstacle tests
|-- benchmarks/
|   `-- bench_physics.py     # Performance throughput benchmarks
|-- examples/
|   `-- physics_lab.py       # Interactive terminal physics laboratory
|-- PLAN.md                  # Complete mathematical and architectural plan
`-- README.md                # System documentation
```

---

## Mathematical Foundations

### 1. Symplectic Euler Integration
For rigid body with mass $m$, moment of inertia $I$, position $x$, angle $\theta$, linear velocity $v$, and angular velocity $\omega$:

$$v_{t + \Delta t} = v_t + \left(\frac{F}{m} + g\right) \Delta t$$

$$\omega_{t + \Delta t} = \omega_t + \left(\frac{\tau}{I}\right) \Delta t$$

$$x_{t + \Delta t} = x_t + v_{t + \Delta t} \Delta t$$

$$\theta_{t + \Delta t} = \theta_t + \omega_{t + \Delta t} \Delta t$$

Updating velocity before position preserves the symplectic 2-form $\mathrm{d}p \wedge \mathrm{d}q$, preventing non-physical energy drift.

### 2. Minkowski Difference & Support Mappings
For convex shapes $A$ and $B$, the Minkowski difference is:

$$A \ominus B = \{a - b \mid a \in A, b \in B\}$$

Intersection holds if and only if $(0, 0) \in A \ominus B$. The support function in direction $d$ is:

$$S_{A \ominus B}(d) = S_A(d) - S_B(-d)$$

### 3. Sequential Impulse Constraint Resolution
For two contacting bodies $A$ and $B$ with relative contact point velocity $v_{rel} = (v_B + \omega_B \times r_B) - (v_A + \omega_A \times r_A)$:

- **Normal Impulse** $P_n = J_n n$:
  $$J_n = \frac{-(v_{rel} \cdot n - v_{bias} - \text{Baumgarte})}{m_A^{-1} + m_B^{-1} + \frac{(r_A \times n)^2}{I_A} + \frac{(r_B \times n)^2}{I_B}}$$
  Clamped such that accumulated normal impulse $\lambda_n \ge 0$.

- **Baumgarte Stabilization**:
  $$\text{Baumgarte} = \frac{\beta}{\Delta t} \max(0, \text{penetration} - \text{slop})$$
  Gently resolves resting overlap over multiple frames without imparting explosive velocities.

- **Coulomb Friction Impulse** $P_t = J_t t$:
  Tangential direction $t = (-n_y, n_x)$. Clamped to friction cone:
  $$|J_t| \le \mu \lambda_n$$

---

## Quickstart & Code Example

```python
from novaphysics import Vec2, Box, Circle, RigidBody, BodyType, Material, World

# 1. Create simulation world with Earth gravity
world = World(gravity=Vec2(0.0, -9.81), sub_steps=2, velocity_iterations=10)

# 2. Add static floor
ground = RigidBody(Box(20.0, 1.0), position=Vec2(0.0, 0.0), body_type=BodyType.STATIC)
world.add_body(ground)

# 3. Add dynamic bouncing ball
bouncy_mat = Material(density=1.0, restitution=0.8, dynamic_friction=0.3)
ball = RigidBody(Circle(0.5), position=Vec2(0.0, 5.0), material=bouncy_mat)
world.add_body(ball)

# 4. Advance physics simulation
dt = 1.0 / 60.0
for frame in range(120):
    world.step(dt)
    print(f"Frame {frame:3d}: Ball Y = {ball.position.y:.3f} m, Vel Y = {ball.velocity.y:.3f} m/s")
```

---

## Performance Benchmarks

Running `python3 benchmarks/bench_physics.py` on macOS (Apple Silicon):

| Benchmark Component | Workload | Measured Throughput | Performance Note |
| :--- | :--- | :--- | :--- |
| **Vector Math** | Dot, cross, rotate, add | **10,514,568 ops/sec** | Zero-allocation slot vectors |
| **Broadphase BVH** | 100-leaf Dynamic AABB tree queries | **85,537 queries/sec** | Surface-Area Heuristic pruning |
| **GJK + EPA Narrowphase** | Convex OBB intersection + MTV | **40,620 tests/sec** | Sub-millimeter contact normals |
| **Rigid Body World Step** | 40 colliding bodies under gravity | **165 steps/sec** | 2.75x faster than real-time 60fps |
| **PBD Cloth Simulation** | 80 particles, 250 constraints | **588 steps/sec** | ~10x faster than real-time 60fps |

---

## Interactive Physics Laboratory

NovaPhysics includes a built-in terminal visualizer with 5 showcase scenes:

```bash
# Stable 7-box Jenga tower with incoming projectile
python3 examples/physics_lab.py --scene jenga

# Newton's Cradle demonstrating momentum transfer
python3 examples/physics_lab.py --scene cradle

# Chaotic Double Pendulum with coupled revolute joints
python3 examples/physics_lab.py --scene double_pendulum

# 10x8 PBD Cloth fluttering in wind and tearing
python3 examples/physics_lab.py --scene cloth

# Avalanche of mixed polygons sliding down angled ramps
python3 examples/physics_lab.py --scene avalanche

# Run all scenes with live terminal animation
python3 examples/physics_lab.py --all --live
```
