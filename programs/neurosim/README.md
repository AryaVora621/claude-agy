# NeuroSim: Biophysical Electrophysiology & Neural Circuit Studio

NeuroSim is a standalone desktop biophysical electrophysiology workstation and neural circuit laboratory built entirely in Python standard library `tkinter` with zero external dependencies.

## Key Features

- **Conductance-Based Biophysics**: First-principles numerical solution of the 4-variable Hodgkin-Huxley (1952) model of excitable membranes.
- **Unconditionally Stable Integration**: Rush-Larsen (1978) exponential Euler integration for sodium ($m, h$) and potassium ($n$) gating variables, guaranteeing strict $[0, 1]$ bounds.
- **Multi-Compartment Cable Theory**: Rall (1959) dendritic cable model linking soma, basal dendrite, apical trunk, and apical tuft compartments via axial resistance $R_a$.
- **Active Back-Propagating Action Potentials (bAP)**: Dendritic sodium and potassium channels supporting physiological action potential back-propagation from soma to dendritic arbor.
- **Receptor Synaptic Kinetics**: Bi-exponential AMPA excitation ($E_{\text{rev}} = 0\ \text{mV}$), GABA_A inhibition ($E_{\text{rev}} = -70\ \text{mV}$), and Jahr-Stevens voltage-dependent magnesium block for NMDA receptors.
- **Microcircuit Dynamics**: PING (Pyramidal-Interneuron Network Gamma) 40 Hz oscillations and Central Pattern Generator (CPG) bilateral reciprocal inhibition half-center locomotion oscillators.
- **Multi-Channel Oscilloscope & Phase-Plane Analysis**: Live digital oscilloscope plotting $V(t)$, injected current $I(t)$, and gating variables alongside a dynamic $(V, n)$ limit cycle phase-plane attractor.
- **Interactive Patch-Clamp Stimulation**: Interactive mouse click-to-inject stimulation, current clamp sliders, and conductance adjustment.

## Mathematical Formulation

### 1. Membrane Potential Equation
$$C_m \frac{dV}{dt} = I_{\text{inj}} - I_{\text{Na}} - I_{\text{K}} - I_L - I_{\text{syn}} + I_{\text{axial}}$$

### 2. Voltage-Gated Ionic Currents
$$I_{\text{Na}} = \bar{g}_{\text{Na}} m^3 h (V - E_{\text{Na}}), \quad I_{\text{K}} = \bar{g}_{\text{K}} n^4 (V - E_{\text{K}}), \quad I_L = g_L (V - E_L)$$

### 3. Gating Particle Kinetics
$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x = \frac{x_\infty(V) - x}{\tau_x(V)}, \quad x \in \{m, h, n\}$$

### 4. Rush-Larsen Exponential Integration
$$x(t + \Delta t) = x_\infty(V) + (x(t) - x_\infty(V)) \exp\left(-\frac{\Delta t}{\tau_x(V)}\right)$$

### 5. Multi-Compartment Axial Coupling
$$I_{\text{axial}, k} = \sum_{j \in \text{neighbors}} g_{\text{axial}} (V_j - V_k)$$

### 6. NMDA Magnesium Block
$$I_{\text{NMDA}} = g_{\text{NMDA}} B(V) (V - E_{\text{NMDA}}), \quad B(V) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]}{3.57} \exp(-0.062 V)}$$

## Verification & Test Suite

Run the automated 16-test suite:
```bash
python3 programs/neurosim/test_neurosim.py
```

All 16 unit tests verify resting equilibrium (-65 mV), action potential overshoot, refractory periods, anode break excitation, cable attenuation, synaptic conductances, and network oscillations with 100% pass rate.

## Launch Desktop Studio

```bash
python3 programs/neurosim/neurosim.py
```
