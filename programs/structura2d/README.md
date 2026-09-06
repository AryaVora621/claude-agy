# Structura 2D: Finite Element Analysis & Continuum Mechanics Studio

Structura 2D is a standalone desktop engineering analysis laboratory built entirely with Python standard library Tkinter. It implements first principles 2D finite element method (FEA) mechanics for 1D truss pin-jointed structures, 2D Constant Strain Triangles (CST), and 4-node Isoparametric Quadrilaterals (Quad4) integrated using 2x2 Gauss-Legendre quadrature.

Zero external dependencies required. Runs out of the box with standard Python 3.

---

## Engineering Features

1. **Finite Element Formulation:**
   - **2-Node Pin-Jointed Truss:** Internal axial force, normal tensile/compressive stress, local-to-global rotation matrix.
   - **3-Node Constant Strain Triangle (CST):** Closed-form linear displacement strain-displacement matrix B (3x6), constitutive matrix D (plane stress and plane strain), exact area evaluation.
   - **4-Node Isoparametric Quadrilateral (Quad4):** Bilinear natural coordinate shape functions, coordinate Jacobian matrix J(xi, eta), 2x2 Gauss-Legendre numerical quadrature integration with 4 evaluation points.

2. **Numerical Solvers:**
   - Global stiffness matrix assembly K and external load vector F.
   - Dirichlet boundary condition partitioning (fixed and free degrees of freedom).
   - Reduced linear system solver via Gaussian elimination with scaled partial pivoting.
   - Nodal reaction force vector recovery at all constrained DOFs.
   - Element stress recovery: sigma_xx, sigma_yy, shear stress tau_xy, principal stresses sigma_1 and sigma_2.
   - Von Mises equivalent failure yield stress and structural safety factor SF.
   - Rayleigh quotient lumped mass fundamental vibration frequency estimation.

3. **Curated Structural Benchmarks:**
   - **Warren-Pratt Bridge Truss:** 7-bay truss with pin and roller supports under simulated vehicular traffic loads.
   - **Cantilever Beam (CST):** Slender flexural beam verifying Euler-Bernoulli deflection mechanics.
   - **Kirsch Plate with Circular Hole (Quad4):** Classical elasticity benchmark demonstrating theoretical stress concentration (Kt approx 3.0) around circular opening.
   - **L-Shaped Structural Bracket (CST):** Discontinuous geometry highlighting re-entrant interior corner stress singularities.
   - **Thick-Walled Cylinder (Quad4):** Internal hydraulic pressure quarter-symmetry model matching analytical Lamé hoop and radial stress distributions.
   - **Multi-Story Shear Wall (Quad4):** High-rise structural wall subject to inverted-triangular lateral wind and seismic shears.

4. **Interactive GUI Capabilities:**
   - Smooth viewport pan and zoom navigation.
   - Real-time contour rendering: Von Mises stress, displacement magnitude, sigma_xx, sigma_yy, tau_xy, and wireframe.
   - Continuous deformation scale factor slider (1x to 1000x).
   - Visual display overlays: Undeformed ghost wireframe, nodal points, external load arrows, support glyphs (pins, rollers), and reaction vectors.
   - Animated fundamental harmonic vibration mode.
   - Click-to-inspect probe displaying nodal coordinates, deformations, reactions, and stresses.

---

## Launch Instructions

```bash
# Launch GUI Studio
python3 programs/structura2d/structura2d.py

# Run Automated Test Suite (16 test cases)
python3 programs/structura2d/test_structura2d.py
```
