# NovaPhysics: 2D Rigid Body Dynamics, GJK/EPA Collision Engine & Constraint Solver
## Technical Specification & Architecture Plan

---

## 1. Executive Overview

NovaPhysics is a production-grade, zero-dependency 2D rigid body physics simulation engine, convex collision detection system, and constraint solver implemented entirely from first principles in the pure Python standard library.

Modern real-time physics engines (like Box2D, PhysX, or Bullet) rely on subtle mathematical and numerical principles to achieve physical plausibility and numerical stability:
1. **Semi-Implicit (Symplectic) Euler Integration**: Preserves phase space energy better than standard forward Euler, preventing artificial energy explosion in oscillatory and gravitational systems.
2. **GJK (Gilbert-Johnson-Keerthi) Distance & Intersection Algorithm**: Computes convex shape intersection in linear/sub-linear time by iteratively enclosing the origin in Minkowski difference space using support mappings.
3. **EPA (Expanding Polytope Algorithm)**: Recursively expands the 2D Minkowski difference simplex to calculate the exact Minimum Translation Vector (MTV), contact normal, and penetration depth.
4. **Sequential Impulse Solver with Warm Starting**: Solves non-linear complementarity problems (LCP) for contact normal impulses and Coulomb friction through projected Gauss-Seidel iterations. Warm starting accumulates impulses across time steps, allowing stable box stacks without jitter or sinking.
5. **Dynamic AABB Bounding Volume Hierarchy (BVH)**: Prunes pairwise collision checks from $O(N^2)$ to $O(N \log N)$ with incremental tree insertions, surface area heuristic rebalancing, and raycasting.
6. **Verlet Cloth & Particle Mechanics**: Position-based dynamics (PBD) with distance relaxation constraints, simulating structural, shear, and tearable cloth meshes.
7. **ANSI Terminal Physics Rasterizer**: Real-time ASCII and TrueColor terminal renderer with sub-pixel braille rasterization, displaying live falling dominoes, stable towers, pendulum double-hinges, and tearing cloth.

---

## 2. Mathematical Foundations

### 2.1 Symplectic Euler Integration
For rigid body with mass $m$, moment of inertia $I$, position $x$, angle $\theta$, linear velocity $v$, and angular velocity $\omega$:

$$v_{t + \Delta t} = v_t + \left(\frac{F}{m} + g\right) \Delta t$$

$$\omega_{t + \Delta t} = \omega_t + \left(\frac{\tau}{I}\right) \Delta t$$

$$x_{t + \Delta t} = x_t + v_{t + \Delta t} \Delta t$$

$$\theta_{t + \Delta t} = \theta_t + \omega_{t + \Delta t} \Delta t$$

Updating velocity before position preserves symplectic structure, preventing spurious energy accumulation.

### 2.2 Support Mappings & Minkowski Difference
For convex shapes $A$ and $B$, the Minkowski difference is:

$$A \ominus B = \{a - b \mid a \in A, b \in B\}$$

Shapes $A$ and $B$ intersect if and only if the origin $O = (0, 0)$ lies inside $A \ominus B$.
The support function $S_A(d)$ returns the vertex of $A$ furthest in direction $d$:

$$S_A(d) = \arg\max_{a \in A} (a \cdot d)$$

The support of the Minkowski difference is calculated without explicitly computing the polygon:

$$S_{A \ominus B}(d) = S_A(d) - S_B(-d)$$

### 2.3 GJK & EPA Algorithms
- **GJK**: Builds an evolving simplex (point, line segment, or triangle) in $A \ominus B$. If the simplex encloses $(0,0)$, an intersection is proven.
- **EPA**: When GJK confirms collision, EPA takes the 2D simplex (triangle enclosing origin) and searches for the closest edge to the origin. It finds a new support point in the edge normal direction and inserts it into the polygon until the distance converges within tolerance $\epsilon$. The closest edge yields the exact penetration depth and contact normal.

### 2.4 Sequential Impulse Resolution
For two contacting bodies $A$ and $B$ with relative contact point velocity $v_{rel} = (v_B + \omega_B \times r_B) - (v_A + \omega_A \times r_A)$:

1. **Normal Impulse** $P_n = J_n n$:
   $$J_n = \frac{-(1 + e) (v_{rel} \cdot n) + \text{Baumgarte}}{m_A^{-1} + m_B^{-1} + \frac{(r_A \times n)^2}{I_A} + \frac{(r_B \times n)^2}{I_B}}$$
   Clamped such that accumulated normal impulse $\lambda_n \ge 0$ (bodies can push, never pull).

2. **Baumgarte Stabilization**:
   $$\text{Baumgarte} = \frac{\beta}{\Delta t} \max(0, \text{penetration} - \text{slop})$$
   Gently resolves resting overlap over multiple frames without imparting explosive velocities.

3. **Coulomb Friction Impulse** $P_t = J_t t$:
   Tangential direction $t = n \times \hat{k}$. Clamped to friction cone:
   $$|J_t| \le \mu J_n$$

4. **Warm Starting**:
   Accumulated impulses from previous frames are applied at the start of each step, providing immediate stacking stability.

---

## 3. Directory Layout & Module Structure

```
projects/12-novaphysics/
|-- novaphysics/
|   |-- __init__.py          # Public package exports
|   |-- math2d.py            # Vec2, Mat22, Transform2D, rotation utilities
|   |-- shapes.py            # Shape, Circle, Polygon, Box, AABB
|   |-- body.py              # RigidBody, BodyType (Dynamic, Static, Kinematic)
|   |-- broadphase.py        # Dynamic AABB Tree with raycasting
|   |-- gjk.py               # Gilbert-Johnson-Keerthi convex collision test
|   |-- epa.py               # Expanding Polytope Algorithm for contact manifolds
|   |-- contact.py           # ContactManifold, ContactPoint, feature clipping
|   |-- solver.py            # SequentialImpulseSolver, Baumgarte, friction
|   |-- joints.py            # DistanceJoint, RevoluteJoint, SpringJoint
|   |-- world.py             # World simulation manager, sub-stepping
|   |-- cloth.py             # Verlet particle system, distance constraints
|   `-- visualizer.py        # ANSI terminal physics canvas and renderer
|-- tests/
|   |-- test_math2d.py       # Vector, matrix, and transform tests
|   |-- test_shapes.py       # Convex polygon vertices and inertia tests
|   |-- test_gjk_epa.py      # Collision detection and penetration tests
|   |-- test_solver.py       # Momentum conservation and stacking tests
|   |-- test_joints.py       # Distance and revolute constraint tests
|   |-- test_cloth.py        # Verlet particle and cloth mesh tests
|   `-- test_broadphase.py   # Dynamic AABB tree queries and pruning
|-- benchmarks/
|   `-- bench_physics.py     # Simulation step rate, GJK ops/s, broadphase ops/s
|-- examples/
|   `-- physics_lab.py       # Interactive terminal physics showcase
|-- PLAN.md                  # System architecture specification
`-- README.md                # System documentation
```

---

## 4. Implementation Steps

1. **Step 1**: Implement `novaphysics/math2d.py` (Vec2, Mat22, Transform2D).
2. **Step 2**: Implement `novaphysics/shapes.py` (Circle, Polygon, Box, AABB).
3. **Step 3**: Implement `novaphysics/body.py` (RigidBody, mass, inertia calculation).
4. **Step 4**: Implement `novaphysics/broadphase.py` (Dynamic AABB Tree).
5. **Step 5**: Implement `novaphysics/gjk.py` & `novaphysics/epa.py` (GJK & EPA narrowphase).
6. **Step 6**: Implement `novaphysics/contact.py` & `novaphysics/solver.py` (impulse solver, warm starting).
7. **Step 7**: Implement `novaphysics/joints.py` & `novaphysics/world.py` (constraints and simulation loop).
8. **Step 8**: Implement `novaphysics/cloth.py` (Verlet particle & cloth dynamics).
9. **Step 9**: Implement `novaphysics/visualizer.py` (ANSI terminal vector canvas).
10. **Step 10**: Build complete unit test suite, benchmark suite, and interactive showcase demo.
11. **Step 11**: Update `showcase.py`, `projects.md`, and master tracker `tracker/data.json`.
