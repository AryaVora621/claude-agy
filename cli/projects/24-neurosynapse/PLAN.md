# PLAN: Project 24 - NeuroSynapse
## Neuromorphic Computing, Spiking Neural Network (SNN) & Event-Based Vision Engine

### Architectural Identity & Philosophy
NeuroSynapse is a zero-dependency, first-principles neuromorphic computing engine and Spiking Neural Network (SNN) simulator written in the pure Python standard library. It bridges biophysical neurodynamics (Hodgkin-Huxley, Izhikevich, Leaky Integrate-and-Fire) with modern neuromorphic machine learning (Surrogate Gradient Backpropagation Through Time, Spike-Timing-Dependent Plasticity) and asynchronous event-driven Dynamic Vision Sensor (DVS) processing.

### Design Principles
1. Zero External Dependencies: Pure Python standard library (`math`, `typing`, `collections`, `random`, `struct`, `time`, `dataclasses`, `heapq`).
2. Mathematical Rigor: Exact numerical integrators (Euler-Maruyama, 4th-order Runge-Kutta), biophysically accurate gating kinetics, and analytical surrogate gradient derivatives.
3. Dual-Mode Execution:
   - Synchronous $\Delta t$ Time-Stepped Tensor Execution: Ideal for batched training and surrogate gradient backpropagation.
   - Asynchronous Event-Driven Priority Queue: Event-triggered synaptic updates executing strictly when action potentials fire, scaling with $O(\text{spikes})$ rather than $O(N^2)$ dense matrix multiplication.
4. Comprehensive Neuromorphic Vision: Full DVS Address-Event Representation (AER) pipeline, refractory filtering, Surface of Active Events (SAE / Time Surface), and event-based optical flow estimation.
5. High-Resolution Visual Telemetry: Sub-pixel Unicode Braille spike raster plots, dual-trace membrane potential oscilloscopes, and DVS visual event accumulator displays.
6. Strict Cleanliness: No em dashes anywhere in code, comments, or documentation.

---

### Module Hierarchy & Architecture

```
neurosynapse/
├── __init__.py          # Public package namespace and version metadata
├── neurons.py           # Biophysical & phenomenological neuron models (LIF, Izhikevich, Hodgkin-Huxley)
├── synapses.py          # Synaptic dynamics (exponential/alpha) and STDP plasticity rules
├── learning.py          # Surrogate gradient backpropagation through time (Spike-BPTT) & autograd
├── dvs.py               # Neuromorphic event-based vision (AER, SAE, filters, optical flow)
├── encoding.py          # Spike encoders (Poisson, TTFS, rank-order, phase) and decoders
├── network.py           # Layer abstractions (Dense, Conv2D, Recurrent LSM) & event queue
├── visualizer.py        # Sub-pixel Braille spike raster, dual oscilloscope, DVS canvas
└── engine.py            # High-level unified orchestrator and 4 flagship presets
```

---

### Mathematical Formulations

#### 1. Neuron Models (`neurosynapse/neurons.py`)
- **Leaky Integrate-and-Fire (LIF) with Adaptive Threshold**:
  $$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$
  $$V_{th}(t) = V_{th,0} + \theta(t), \quad \tau_\theta \frac{d\theta}{dt} = -\theta$$
  If $V(t) \ge V_{th}(t)$, an action potential $S(t) = 1$ is emitted, $V$ is clamped to $V_{reset}$ for refractory duration $\tau_{ref}$, and $\theta \leftarrow \theta + \Delta \theta$.

- **Izhikevich Phenomenological 2D System**:
  $$\frac{dv}{dt} = 0.04 v^2 + 5v + 140 - u + I$$
  $$\frac{du}{dt} = a(bv - u)$$
  Spike condition: if $v \ge 30 \text{ mV}$, then $v \leftarrow c$ and $u \leftarrow u + d$.
  Supports 8 canonical cortical firing patterns:
  - Regular Spiking (RS): $a=0.02, b=0.2, c=-65, d=8$
  - Intrinsically Bursting (IB): $a=0.02, b=0.2, c=-55, d=4$
  - Chattering (CH): $a=0.02, b=0.2, c=-50, d=2$
  - Fast Spiking (FS): $a=0.1, b=0.2, c=-65, d=2$
  - Low-Threshold Spiking (LTS): $a=0.02, b=0.25, c=-65, d=2$
  - Thalamo-Cortical (TC): $a=0.02, b=0.25, c=-65, d=0.05$
  - Resonator (RZ): $a=0.1, b=0.26, c=-65, d=2$
  - Accommodating (ACC): $a=0.02, b=1.0, c=-55, d=4$

- **Hodgkin-Huxley 4-State Conductance Model (1952)**:
  $$\frac{dV}{dt} = \frac{1}{C_m} \left( I_{inj} - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_K n^4 (V - E_K) - \bar{g}_L (V - E_L) \right)$$
  Gating variable dynamics:
  $$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x, \quad x \in \{m, h, n\}$$
  Integrated using 4th-Order Runge-Kutta (RK4) for numerical stability.

#### 2. Synaptic Dynamics & Plasticity (`neurosynapse/synapses.py`)
- Post-Synaptic Current (PSC):
  - Exponential PSC: $I_{syn}(t) = \sum_k w_k e^{-(t - t_k)/\tau_{syn}}$
  - Alpha-Function PSC: $I_{syn}(t) = \sum_k w_k \frac{t - t_k}{\tau_{syn}} e^{-(t - t_k)/\tau_{syn}}$
- Asymmetric Pair-Based STDP:
  $$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+}, & \Delta t = t_{post} - t_{pre} > 0 \quad (\text{LTP}) \\ -A_- e^{\Delta t / \tau_-}, & \Delta t = t_{post} - t_{pre} < 0 \quad (\text{LTD}) \end{cases}$$
- Triplet STDP (Pfister-Gerstner model): incorporates pre-post-post and post-pre-pre spike combinations for biological frequency sensitivity.
- Homeostatic Weight Normalization: multi-synaptic multiplicative scaling $\sum_j w_j = W_{target}$ to prevent runaway excitation.

#### 3. Surrogate Gradient Learning (Spike-BPTT) (`neurosynapse/learning.py`)
- Forward spike generation: $S = \Theta(V - V_{th})$ (non-differentiable Heaviside step).
- Backward surrogate derivatives:
  - FastSigmoid: $\frac{dS}{dV} = \frac{1}{(1 + k|V - V_{th}|)^2}$
  - ArcTan: $\frac{dS}{dV} = \frac{1}{\pi (1 + (k(V - V_{th}))^2)}$
  - Triangular: $\frac{dS}{dV} = \max\left(0, 1 - \frac{|V - V_{th}|}{\gamma}\right) \cdot \frac{1}{\gamma}$
- Temporal autograd backward pass through unrolled simulation steps $t = 1 \dots T$.
- Optimizers: Pure Python SGD with momentum and Adam.

#### 4. Neuromorphic Event Vision (DVS) (`neurosynapse/dvs.py`)
- Address-Event Representation: tuple $(x, y, t, p)$ with polarity $p \in \{-1, +1\}$.
- Spatio-temporal event filters:
  - Refractory filter: drops events occurring within $\Delta t < \tau_{ref}$ for the same pixel.
  - Background Activity Filter (BAF): rejects isolated events with no spatial neighbors within temporal window $\Delta T$.
- Surface of Active Events (SAE / Time Surface):
  $$\Sigma(x, y, p) = t_{last}(x, y, p)$$
- Event-Based Optical Flow:
  Local planar fitting to $\Sigma(x, y)$ in a $k \times k$ patch: $a x + b y + c = t$, yielding gradient $(a, b)$ and normal velocity $\mathbf{v} = \frac{(a, b)}{a^2 + b^2}$.

#### 5. Spike Encoders and Decoders (`neurosynapse/encoding.py`)
- Encoders:
  - Poisson Rate Coding: spike probability $P(\text{spike}) = r \cdot \Delta t$.
  - Bernoulli Rate Coding: uniform random comparison against normalized intensity.
  - Time-to-First-Spike (TTFS): latency $t_{spike} = \tau \ln\left(\frac{1}{1 - x}\right)$.
  - Rank Order Coding (ROC): sequence ordering based on stimulus magnitude.
  - Phase Coding: spike timing locked to reference oscillatory cycle.
- Decoders:
  - Membrane voltage readout: $y = \frac{1}{T} \sum_{t=1}^T V(t)$ or final $V(T)$.
  - Spike count / firing rate: $y = \frac{1}{T} \sum_{t=1}^T S(t)$.
  - Exponential filtering: low-pass continuous trace filtering.

#### 6. Network Topologies & Event-Driven Engine (`neurosynapse/network.py`)
- Layers:
  - `DenseSpikingLayer`: fully-connected weights $W \in \mathbb{R}^{M \times N}$.
  - `Conv2DSpikingLayer`: spatial convolution with stride and padding.
  - `RecurrentSpikingLayer` (Liquid State Machine): recurrent connection matrix $W_{rec}$ with Dale's principle (80% excitatory, 20% inhibitory) and spectral radius scaling.
- Execution Engines:
  - Synchronous Matrix Mode: fixed $\Delta t$ tensor steps.
  - Sparse Event-Queue Engine: min-heap priority queue processing spikes asynchronously at event timestamps, only propagating along connected outgoing synapses.

#### 7. Unicode Braille Visualizer (`neurosynapse/visualizer.py`)
- Sub-pixel Braille spike raster: 2x4 subpixels per character cell, displaying up to 64 neurons over 100 time steps cleanly on a standard terminal.
- Oscilloscope: dual-trace voltage plot ($V_m$ and adaptive $V_{th}$) with ANSI color coding and threshold annotations.
- DVS Event Canvas: 2D sub-pixel rendering of event clouds (green for ON events, red for OFF events).

---

### Verification and Test Plan
- Unit tests covering all 8 modules (`tests/test_neurons.py`, `tests/test_synapses_stdp.py`, `tests/test_learning.py`, `tests/test_dvs.py`, `tests/test_encoding.py`, `tests/test_network.py`, `tests/test_visualizer.py`).
- Target: 25+ comprehensive unit tests, 100% passing.
- Microbenchmarks (`benchmarks/bench_neurosynapse.py`):
  - LIF, Izhikevich, Hodgkin-Huxley integration throughput (ops/sec).
  - STDP weight update speed (synapses/sec).
  - Surrogate gradient Spike-BPTT backward pass throughput (timesteps/sec).
  - Event-queue vs synchronous simulation throughput comparison.
  - DVS event filtering and SAE generation rate (events/sec).
- Flagship Workbench (`examples/neuromorphic_workbench.py`):
  - Audio Spiking Spectrogram Classifier.
  - Liquid State Machine Chaos Classifier.
  - Unsupervised STDP Feature Discovery.
  - DVS Optical Flow Motion Tracker.
