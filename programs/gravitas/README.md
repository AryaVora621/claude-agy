# Gravitas 3D - Standalone Desktop N-Body Orbital Mechanics Studio

Gravitas 3D is an interactive, high-precision N-body gravitational physics laboratory and celestial mechanics simulator built entirely with the Python standard library (`tkinter`). It features real-time 3D perspective projection, symplectic integrators, post-Newtonian relativistic corrections, and interactive celestial body creation.

---

## 🌟 Key Features

- **3D Perspective Projection Engine**:
  - Full spherical orbital camera with real-time azimuth orbit, elevation tilt, and distance zooming.
  - Interactive right-click camera target panning across the view plane.
  - Coordinate reference grid plane with concentric distance rings and elevation drop-lines.
  - Dynamic 3D depth-sorting so distant bodies, trails, and sparks render realistically behind nearer objects.
- **Symplectic Numerical Integrators**:
  - **4th-Order Symplectic Yoshida Integrator**: High-precision composition method ensuring near-zero Hamiltonian energy drift ($\Delta E / E_0 < 10^{-4}$) over thousands of orbital periods.
  - **Symplectic Velocity Verlet**: Classic energy-conserving second-order molecular and orbital integrator.
- **Post-Newtonian Relativistic Precession (1PN)**:
  - Toggleable effective potential correction ($\sim 3GM/c^2r$) producing authentic Schwarzschild perihelion precession (rosette orbits) without breaking conservation laws.
- **Inelastic Collision & Coalescence Dynamics**:
  - Physical collision detection preserving total mass, center of mass, and linear momentum ($m_1 \mathbf{v}_1 + m_2 \mathbf{v}_2 = M_{tot} \mathbf{v}_{new}$).
  - Volume conservation ($R_{new} = \sqrt[3]{R_1^3 + R_2^3}$) and kinetic energy dissipation into expanding spark particle explosions.
- **Interactive Body Sling Mode**:
  - Click and drag anywhere on the 3D grid plane to draw a launch velocity vector arrow.
  - Adjust mass, radius, and color, then release to fling new asteroids, comets, or rogue planets into active orbit.
- **Curated Celestial Choreography Presets**:
  1. **Figure-8 Choreography**: The famous 3-body equal-mass solution (Chenciner & Montgomery 2000) where three bodies chase each other along a figure-eight loop.
  2. **Sol System Core**: Sun, Mercury, Venus, Earth, Mars, Jupiter, and Saturn scaled with authentic mass ratios, inclinations, and orbital velocities.
  3. **Lagrangian Trojan Resonance**: Primary star, jovian gas giant, and stable swarms of L4 / L5 Trojan asteroids in 1:1 orbital resonance.
  4. **Pythagorean 3-Body Problem**: Burrau's classical 1913 problem demonstrating chaotic close encounters and temporary binary formation.
  5. **Galactic Disk Merger**: Two colliding spiral galaxies with central black holes and orbiting star rings exhibiting tidal tails and bridges.
  6. **Relativistic Rosette**: Supermassive black hole with an eccentric star experiencing relativistic perihelion advance.
- **Live Hamiltonian Telemetry HUD**:
  - Real-time readout of Kinetic Energy ($T$), Potential Energy ($U$), Total Energy ($E$), and fractional Hamiltonian drift ($\Delta E / E_0$).
  - Angular momentum vector magnitude ($|\mathbf{L}|$).
  - Body Inspector Pane: Select any body to lock the tracking camera, view instantaneous velocity, mass, and distance.

---

## 🚀 Quick Start

Launch the desktop GUI directly from terminal:

```bash
python3 programs/gravitas/gravitas.py
```

Run the automated test suite:

```bash
python3 -m unittest programs/gravitas/test_gravitas.py
```

---

## 🎮 Controls

- **Left Mouse Drag**: Orbit 3D camera (rotate azimuth and elevation tilt).
- **Right Mouse Drag**: Pan camera focus target.
- **Mouse Wheel / Trackpad Pinch**: Smooth zoom in/out.
- **Left Click on Body**: Select body to inspect real-time orbital metrics and lock tracking camera.
- **Sling Mode (Toggle Button)**: Click & drag on the 3D grid plane to aim and launch a new planet into orbit.
