# Architectural Specification: Project 32 (ThermoProp)

## Executive Summary

**ThermoProp** is a first-principles, zero-dependency compressible gas dynamics simulation engine, supersonic De Laval nozzle design suite via the Method of Characteristics (MOC), thermochemical rocket propulsion performance analyzer, and regenerative thrust chamber cooling laboratory implemented in pure Python 3.10+ standard library.

The engine solves the non-linear 1D/2D compressible Navier-Stokes and Euler equations, evaluates isentropic flow relations and Rankine-Hugoniot shock jumps, designs shock-free minimum-length supersonic nozzle contours via the 2D Method of Characteristics (MOC), calculates rocket engine combustion metrics (characteristic velocity $c^*$, thrust coefficient $C_F$, vacuum and sea-level specific impulse $I_{sp}$), integrates Bartz convective heat transfer profiles with 1D regenerative cooling thermal resistance networks, models supersonic exhaust plume shock reflections (Mach diamonds) across overexpanded, underexpanded, and adapted regimes, and renders sub-pixel Unicode Braille supersonic contours with 24-bit TrueColor aerothermodynamic telemetry.

---

## 1. Compressible Gas Dynamics & Aerothermodynamics

### 1.1 Isentropic Compressible Flow
For an ideal gas with ratio of specific heats $\gamma = c_p / c_v$ and gas constant $R_{gas}$:
Stagnation temperature $T_0$ and stagnation pressure $P_0$ relate to local static conditions $(T, P, \rho)$ and local Mach number $M = v / a$ (where acoustic speed $a = \sqrt{\gamma R_{gas} T}$):
$$\frac{T_0}{T} = 1 + \frac{\gamma - 1}{2} M^2$$
$$\frac{P_0}{P} = \left( 1 + \frac{\gamma - 1}{2} M^2 \right)^{\frac{\gamma}{\gamma - 1}}$$
$$\frac{\rho_0}{\rho} = \left( 1 + \frac{\gamma - 1}{2} M^2 \right)^{\frac{1}{\gamma - 1}}$$

### 1.2 The Area-Mach Number Relation
At the choked sonic throat, $M = 1$ and cross-sectional area is $A^* = A_t$. The local cross-sectional area ratio $A / A^*$ satisfies:
$$\frac{A}{A^*} = \frac{1}{M} \left[ \frac{2}{\gamma + 1} \left( 1 + \frac{\gamma - 1}{2} M^2 \right) \right]^{\frac{\gamma + 1}{2(\gamma - 1)}}$$

The engine solves for Mach number $M(A/A^*)$ using high-order Halley root finding for both subsonic ($M < 1$) converging and supersonic ($M > 1$) diverging branches.

### 1.3 Normal & Oblique Shock Waves (Rankine-Hugoniot)
Across a normal shock wave with upstream Mach number $M_1 > 1$:
$$M_2 = \sqrt{\frac{2 + (\gamma - 1) M_1^2}{2 \gamma M_1^2 - (\gamma - 1)}}$$
$$\frac{P_2}{P_1} = 1 + \frac{2\gamma}{\gamma + 1} (M_1^2 - 1)$$
$$\frac{T_2}{T_1} = \frac{P_2}{P_1} \left( \frac{2 + (\gamma - 1) M_1^2}{(\gamma + 1) M_1^2} \right)$$
$$\frac{P_{02}}{P_{01}} = \exp\left(-\frac{\Delta s}{R_{gas}}\right)$$

For an oblique shock with wave angle $\beta$ and flow deflection angle $\theta$:
$$\tan\theta = 2 \cot\beta \left[ \frac{M_1^2 \sin^2\beta - 1}{M_1^2 (\gamma + \cos(2\beta)) + 2} \right]$$

### 1.4 Prandtl-Meyer Supersonic Expansion
For supersonic expansion around a convex corner:
$$\nu(M) = \sqrt{\frac{\gamma + 1}{\gamma - 1}} \arctan\left(\sqrt{\frac{\gamma - 1}{\gamma + 1} (M^2 - 1)}\right) - \arctan\left(\sqrt{M^2 - 1}\right)$$
Prandtl-Meyer angle $\nu(M)$ is invertible via analytical and iterative approximations.

---

## 2. 2D Method of Characteristics (MOC) Supersonic Nozzle Design

### 2.1 Characteristic Equations in 2D Irrotational Supersonic Flow
In supersonic flow ($M > 1$), the steady 2D potential flow equation is hyperbolic with two families of real characteristic Mach lines ($C^+$ and $C^-$) inclined at the Mach angle $\mu = \arcsin(1/M)$:
$$C^+ : \frac{dy}{dx} = \tan(\theta + \mu), \quad \theta + \nu(M) = K^+ = \text{const}$$
$$C^- : \frac{dy}{dx} = \tan(\theta - \mu), \quad \theta - \nu(M) = K^- = \text{const}$$

where $\theta$ is the local streamline flow angle and $\nu(M)$ is the Prandtl-Meyer angle.

### 2.2 Minimum-Length Nozzle (MLN) Synthesis
1. Expansion fan generated at the sharp or rounded throat corner until maximum turning angle $\theta_{max} = \frac{1}{2} \nu(M_{exit})$.
2. Characteristic mesh intersection: interior points computed by solving Riemann invariants $K^+$ and $K^-$.
3. Straightening section: cancel expansion waves to yield uniform, parallel, axial exit flow ($\theta = 0, M = M_{exit}$) at the nozzle lip, minimizing divergence losses.

---

## 3. Rocket Thermochemistry & Propulsion Performance

### 3.1 Ideal Rocket Performance Parameters
- **Thrust**: $F = \dot{m} v_e + (P_e - P_a) A_e$
- **Characteristic Velocity**:
  $$c^* = \frac{P_c A_t}{\dot{m}} = \frac{\sqrt{\gamma R_{gas} T_c}}{\gamma \sqrt{\left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma + 1}{\gamma - 1}}}}$$
- **Thrust Coefficient**:
  $$C_F = \sqrt{\frac{2 \gamma^2}{\gamma - 1} \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma + 1}{\gamma - 1}} \left[ 1 - \left(\frac{P_e}{P_c}\right)^{\frac{\gamma - 1}{\gamma}} \right]} + \frac{P_e - P_a}{P_c} \frac{A_e}{A_t}$$
- **Specific Impulse**:
  $$I_{sp} = \frac{F}{\dot{m} g_0} = \frac{c^* C_F}{g_0}$$

### 3.2 Standard Propellant Presets
- **Hydrolox ($LOX / LH_2$)**: $T_c \approx 3600 \text{ K}, \gamma \approx 1.22, M_w \approx 13.5 \text{ g/mol}, I_{sp,vac} \approx 450 \text{ s}$
- **Methalox ($LOX / LCH_4$)**: $T_c \approx 3550 \text{ K}, \gamma \approx 1.20, M_w \approx 20.0 \text{ g/mol}, I_{sp,vac} \approx 380 \text{ s}$
- **Kerolox ($LOX / RP\text{-}1$)**: $T_c \approx 3700 \text{ K}, \gamma \approx 1.24, M_w \approx 23.5 \text{ g/mol}, I_{sp,vac} \approx 340 \text{ s}$
- **Storable Hypergolic ($N_2O_4 / UDMH$)**: $T_c \approx 3400 \text{ K}, \gamma \approx 1.25, M_w \approx 21.0 \text{ g/mol}, I_{sp,vac} \approx 320 \text{ s}$

---

## 4. Regenerative Thrust Chamber Cooling & Heat Flux Analysis

### 4.1 Bartz Empirical Equation for Gas-Side Heat Transfer
The convective heat transfer coefficient $h_g(x)$ along the chamber and nozzle is:
$$h_g = \frac{0.026}{D_t^{0.2}} \left( \frac{\mu^{0.2} c_p}{Pr^{0.6}} \right)_{core} \left( \frac{P_c}{c^*} \right)^{0.8} \left( \frac{D_t}{r_c} \right)^{0.1} \left( \frac{A_t}{A} \right)^{0.9} \sigma$$
where $\sigma$ is the boundary layer correction factor:
$$\sigma = \left[ \frac{1}{2} \frac{T_w}{T_0} \left( 1 + \frac{\gamma - 1}{2} M^2 \right) + \frac{1}{2} \right]^{-0.68} \left[ 1 + \frac{\gamma - 1}{2} M^2 \right]^{-0.12}$$

### 4.2 1D Thermal Resistance Network
Heat flux through nozzle wall into regenerative cooling jacket:
$$q = \frac{T_{aw} - T_{coolant}}{\frac{1}{h_g} + \frac{t_{wall}}{k_{wall}} + \frac{1}{h_{coolant}}}$$
where adiabatic recovery temperature $T_{aw} = T (1 + r \frac{\gamma - 1}{2} M^2)$ with recovery factor $r \approx Pr^{1/3}$.

---

## 5. Supersonic Exhaust Plumes & Mach Diamonds

### 5.1 Pressure Matching Regimes
- **Under-expanded ($P_e > P_a$)**: Flow expands further via Prandtl-Meyer expansion fans at nozzle lip.
- **Ideally expanded ($P_e = P_a$)**: Parallel cylindrical plume with maximum thrust coefficient.
- **Over-expanded ($P_e < P_a$)**: Oblique compression shock waves originate at the nozzle lip, reflecting at the jet centerline to produce periodic shock cells (Mach diamonds).
- **Summerfield Criterion**: Flow separation occurs when $P_e / P_a \lesssim 0.35 - 0.40$.

---

## 6. Sub-Pixel Braille Visualizer & Telemetry HUD

- **2x4 Sub-Pixel Braille Canvas**: Renders high-resolution nozzle contour $y(x)$, sonic throat line, Mach lines, and supersonic exhaust plume.
- **24-bit TrueColor ANSI Gradient**:
  - Subsonic chamber: Hot white-red ($T > 3000$ K)
  - Sonic throat: Intense amber ($M = 1.0$)
  - Supersonic expansion: Bright cyan/blue ($M > 2.0$)
  - Shock cells & Mach diamonds: Vivid neon green/yellow highlights.
- **Rocket Propulsion Telemetry HUD**:
  - $P_c, T_c, \dot{m}, F_{vac}, F_{sl}, I_{sp,vac}, I_{sp,sl}, c^*, C_F, \epsilon = A_e/A_t$.

---

## 7. Directory Layout & Module Structure

```
projects/32-thermoprop/
├── thermoprop/
│   ├── __init__.py          # Public API exports
│   ├── gas_dynamics.py      # Isentropic flow, normal/oblique shocks, Prandtl-Meyer
│   ├── moc_nozzle.py        # 2D Method of Characteristics supersonic nozzle design
│   ├── propulsion.py        # Rocket performance, thermochemistry presets, Isp, thrust
│   ├── cooling.py           # Bartz heat transfer, thermal resistance, regenerative channels
│   ├── plume.py             # Exhaust plume shock cells, Mach diamonds & separation
│   └── visualizer.py        # Sub-pixel Unicode Braille nozzle/plume visualizer & HUD
├── tests/
│   ├── __init__.py
│   ├── test_gas_dynamics.py # Isentropic relations, Area-Mach, shock jump conditions
│   ├── test_moc_nozzle.py   # Characteristic invariants, exit Mach uniformity, contour
│   ├── test_propulsion.py   # c*, CF, Isp, vacuum vs sea-level thrust
│   ├── test_cooling.py      # Bartz coefficient, wall temperatures, coolant heat uptake
│   ├── test_plume.py        # Shock cell spacing, pressure ratio regimes
│   └── test_visualizer.py   # Braille canvas rasterization, color scales, HUD formatting
├── benchmarks/
│   └── bench_thermoprop.py  # Throughput microbenchmarks (gas evals/s, MOC nodes/s, etc.)
├── examples/
│   └── rocket_workbench.py  # Interactive CLI rocket propulsion laboratory
├── PLAN.md                  # Comprehensive architectural specification
├── TASK_QUEUE.md            # Milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # System documentation
```
