# KineMatix 3D: Robotics & Multibody Inverse Kinematics Studio

Standalone desktop robotics laboratory and multibody kinematics workstation built in pure Python standard library `tkinter` with zero external pip dependencies.

---

## Technical Architecture & Mathematical Formulations

KineMatix 3D provides an exact mathematical implementation of serial kinematic chains and parallel robotic platforms from first principles.

### 1. Denavit-Hartenberg (DH) Homogeneous Transforms

Each link in a serial kinematic chain is defined by four geometric parameters:
- $\theta_i$: Joint angle around $z_{i-1}$
- $d_i$: Link offset along $z_{i-1}$
- $a_i$: Link length along $x_i$
- $\alpha_i$: Link twist around $x_i$

The coordinate transformation from frame $i-1$ to frame $i$ is given by:

$$T_i^{i-1} = \text{Rot}_z(\theta_i) \text{Trans}_z(d_i) \text{Trans}_x(a_i) \text{Rot}_x(\alpha_i)$$

$$\begin{bmatrix}
\cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\
\sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

Cumulative forward kinematics maps joint state vector $\mathbf{q} \in \mathbb{R}^n$ to end-effector pose:

$$T_n^0(\mathbf{q}) = T_1^0(q_1) T_2^1(q_2) \cdots T_n^{n-1}(q_n)$$

### 2. Geometric Jacobian Derivation

The $6 \times n$ geometric Jacobian matrix $J(\mathbf{q})$ maps joint velocities $\dot{\mathbf{q}}$ to the end-effector spatial twist $\mathbf{v}_e = [\dot{\mathbf{p}}_e^T, \boldsymbol{\omega}_e^T]^T$:

$$\mathbf{v}_e = J(\mathbf{q}) \dot{\mathbf{q}}$$

For each joint $i \in \{1, \dots, n\}$:
- Revolute joint:
  $$J_v^{(i)} = \mathbf{z}_{i-1} \times (\mathbf{p}_e - \mathbf{p}_{i-1}), \quad J_\omega^{(i)} = \mathbf{z}_{i-1}$$
- Prismatic joint:
  $$J_v^{(i)} = \mathbf{z}_{i-1}, \quad J_\omega^{(i)} = \mathbf{0}$$

### 3. Singularity-Robust Damped Least-Squares (DLS) Inverse Kinematics

When the manipulator approaches kinematic singularities (where $\det(J J^T) \to 0$), the standard Moore-Penrose pseudo-inverse $J^\dagger = J^T(J J^T)^{-1}$ causes joint velocities to diverge towards infinity.

KineMatix implements Damped Least-Squares (Levenberg-Marquardt) inverse kinematics:

$$\Delta \mathbf{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}$$

where $\lambda$ is the damping factor, $I$ is the $3 \times 3$ identity matrix, and $\mathbf{e} = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{curr}}$ is the Cartesian position error vector. The $3 \times 3$ linear system $(J J^T + \lambda^2 I) \mathbf{y} = \mathbf{e}$ is solved numerically using Gaussian elimination with partial row pivoting.

### 4. Yoshikawa Manipulability Measure

The dexterity volume of the end-effector is quantified by the Yoshikawa manipulability index:

$$w(\mathbf{q}) = \sqrt{\det(J(\mathbf{q}) J(\mathbf{q})^T)}$$

KineMatix renders the 3D velocity manipulability ellipsoid at the tool center point, visualizing the principal directions and relative mobility of the manipulator in task space.

### 5. Stewart-Gough Parallel Hexapod Inverse Kinematics

For a 6-DOF parallel platform with 6 linear actuator struts connecting fixed base anchor points $\mathbf{b}_i$ to mobile platform joints $\mathbf{p}_i$:

$$\mathbf{l}_i = \mathbf{t} + R(\phi, \theta, \psi) \mathbf{p}_i - \mathbf{b}_i$$

$$L_i = \|\mathbf{l}_i\| = \sqrt{\mathbf{l}_i \cdot \mathbf{l}_i}$$

where $\mathbf{t} = [x, y, z]^T$ is the Cartesian translation vector and $R(\phi, \theta, \psi)$ is the Roll-Pitch-Yaw rotation matrix.

---

## Curated Robotic Presets

1. **PUMA 560 (6-DOF)**: The canonical industrial articulated robot arm with 3-axis spherical wrist and orthogonal shoulder.
2. **UR5 Cobot (6-DOF)**: Modern collaborative robotic arm with cylindrical reach and zero-offset elbow geometry.
3. **SCARA (4-DOF)**: Selective Compliance Assembly Robot Arm with planar articulation and linear vertical Z plunge.
4. **Stanford Arm (6-DOF)**: Historical 1969 manipulator (Scheiman) featuring a prismatic boom extension (RRPRRR).
5. **7-DOF Anthropomorphic Arm**: Kinematically redundant humanoid arm model with shoulder, elbow, and wrist swivel.
6. **Stewart-Gough Platform (6-DOF)**: High-stiffness parallel hexapod with 6 telescopic linear actuators.

---

## Desktop GUI Controls

- **Orbit Camera**: Left-click and drag to rotate azimuth and elevation around the robot workspace.
- **Pan View**: Right-click (or middle-click) and drag to pan camera focus center.
- **Zoom**: Mouse wheel or zoom buttons to scale viewing distance.
- **Direct 3D Target Drag**: Toggle mouse mode to "Drag 3D Target" to position the end-effector target marker directly in the 3D viewport.
- **Joint Sliders**: Manual joint manipulation in degrees or millimeters with physical limit clamping.
- **Trajectory Generator**: Continuous automated tracking across Circle, Figure-8 Lissajous, Square Box, and Helical Spiral paths.
- **Manipulability Ellipsoid**: Real-time 3D wireframe ellipsoid rendering at the tool tip showing dexterity axes.

---

## Launch Command

Run the application directly using standard Python:

```bash
python3 programs/kinematix/kinematix.py
```

Run the automated unit test suite:

```bash
python3 -m unittest programs/kinematix/test_kinematix.py
```
