"""
NeuroSim: Standalone Desktop Biophysical Electrophysiology Studio.
Interactive Hodgkin-Huxley membrane dynamics, cable theory, and neural circuits.
Standard library Python tkinter: zero external dependencies.
"""

import sys
import os
import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.neurosim.biophys_engine import BiophysicalNeuron, NeuralCircuit
from programs.neurosim.presets import PRESETS


class NeuroSimApp:
    """Standalone Desktop GUI for NeuroSim Biophysical Studio."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroSim: Biophysical Electrophysiology & Neural Circuit Studio")
        self.root.geometry("1240x820")
        self.root.minsize(1050, 720)
        self.root.configure(bg="#070B16")

        # Simulation parameters
        self.running = True
        self.sim_speed = 1.0
        self.dt = 0.02  # ms
        self.current_preset_key = "giant_squid"

        # Load initial preset
        self.circuit, self.metadata = PRESETS[self.current_preset_key]()
        self.primary_id = self.metadata.get("primary_id", 0)

        # Telemetry cache
        self.spike_count = 0
        self.firing_rate_hz = 0.0
        self.last_spike_check_time = 0.0

        # Oscilloscope trace buffer
        self.trace_length = 380
        self.trace_v: List[float] = []
        self.trace_i: List[float] = []
        self.trace_m: List[float] = []
        self.trace_h: List[float] = []
        self.trace_n: List[float] = []
        self.trace_v2: List[float] = []

        self._init_ui()
        self._animate()

    def _init_ui(self):
        # Master Style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#070B16")
        style.configure("TLabel", background="#070B16", foreground="#F1F5F9", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"), foreground="#00F0FF")
        style.configure("Sub.TLabel", font=("Helvetica", 9), foreground="#94A3B8")
        style.configure("TButton", background="#131D36", foreground="#F1F5F9", borderwidth=1, font=("Helvetica", 9, "bold"))
        style.map("TButton", background=[("active", "#1E2D52"), ("pressed", "#00F0FF")])

        # Top Control & Preset Toolbar
        top_bar = tk.Frame(self.root, bg="#0C1326", height=48, padx=14, pady=8, highlightbackground="#253556", highlightthickness=1)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        lbl_title = tk.Label(top_bar, text="NEUROSIM", font=("Helvetica", 13, "bold"), fg="#00F0FF", bg="#0C1326")
        lbl_title.pack(side=tk.LEFT, padx=(0, 6))

        lbl_sub = tk.Label(top_bar, text="Biophysical Electrophysiology", font=("Helvetica", 9), fg="#94A3B8", bg="#0C1326")
        lbl_sub.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(top_bar, text="Preset Circuit:", font=("Helvetica", 9, "bold"), fg="#F1F5F9", bg="#0C1326").pack(side=tk.LEFT, padx=(0, 6))
        self.preset_var = tk.StringVar(value=self.current_preset_key)
        preset_menu = ttk.Combobox(
            top_bar,
            textvariable=self.preset_var,
            values=[
                ("giant_squid", "Hodgkin-Huxley Giant Squid Axon"),
                ("anode_break", "Anode Break Excitation (Rebound)"),
                ("ping_gamma", "PING Cortical Gamma Oscillations"),
                ("half_center", "Locomotor Half-Center CPG"),
                ("dendritic", "Dendritic Cable & Back-Propagation"),
                ("thalamic", "Thalamocortical Bursting vs Tonic"),
            ],
            state="readonly",
            width=32,
        )
        preset_menu.set("Hodgkin-Huxley Giant Squid Axon")
        preset_menu.pack(side=tk.LEFT, padx=(0, 14))
        preset_menu.bind("<<ComboboxSelected>>", self._on_preset_change)

        self.btn_pause = tk.Button(top_bar, text="Pause", command=self._toggle_pause, bg="#1E2A4A", fg="#FFF", width=7, relief=tk.FLAT)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        btn_step = tk.Button(top_bar, text="Step", command=self._step_once, bg="#1E2A4A", fg="#FFF", width=6, relief=tk.FLAT)
        btn_step.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(top_bar, text="Reset", command=self._reset_sim, bg="#1E2A4A", fg="#FFF", width=6, relief=tk.FLAT)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Pulse injection quick action
        btn_pulse = tk.Button(top_bar, text="Stim Pulse (15 uA)", command=self._stim_pulse, bg="#006680", fg="#00F0FF", width=14, relief=tk.FLAT)
        btn_pulse.pack(side=tk.RIGHT, padx=6)

        # Main Layout Container
        main_box = tk.Frame(self.root, bg="#070B16")
        main_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Controls Panel
        left_panel = tk.Frame(main_box, bg="#0D152B", width=310, highlightbackground="#253556", highlightthickness=1, padx=14, pady=12)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # 1. Patch-Clamp Injected Current
        tk.Label(left_panel, text="PATCH-CLAMP STIMULATION", font=("Helvetica", 10, "bold"), fg="#00F0FF", bg="#0D152B").pack(anchor=tk.W, pady=(0, 6))

        lbl_inj = tk.Frame(left_panel, bg="#0D152B")
        lbl_inj.pack(fill=tk.X)
        tk.Label(lbl_inj, text="Injected Current (I_inj):", font=("Helvetica", 9), fg="#94A3B8", bg="#0D152B").pack(side=tk.LEFT)
        self.lbl_inj_val = tk.Label(lbl_inj, text="9.5 uA/cm2", font=("Courier", 9, "bold"), fg="#F59E0B", bg="#0D152B")
        self.lbl_inj_val.pack(side=tk.RIGHT)

        self.slider_inj = tk.Scale(
            left_panel,
            from_=-10.0,
            to=30.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_inj_change,
            bg="#0D152B",
            troughcolor="#1A243F",
            fg="#FFF",
            highlightthickness=0,
        )
        self.slider_inj.set(9.5)
        self.slider_inj.pack(fill=tk.X, pady=(2, 10))

        # 2. Membrane Conductances
        tk.Label(left_panel, text="IONIC CONDUCTANCES", font=("Helvetica", 10, "bold"), fg="#00F0FF", bg="#0D152B").pack(anchor=tk.W, pady=(8, 6))

        # g_Na
        lbl_gna = tk.Frame(left_panel, bg="#0D152B")
        lbl_gna.pack(fill=tk.X)
        tk.Label(lbl_gna, text="Sodium (g_Na):", font=("Helvetica", 9), fg="#94A3B8", bg="#0D152B").pack(side=tk.LEFT)
        self.lbl_gna_val = tk.Label(lbl_gna, text="120 mS", font=("Courier", 9, "bold"), fg="#00F0FF", bg="#0D152B")
        self.lbl_gna_val.pack(side=tk.RIGHT)
        self.slider_gna = tk.Scale(
            left_panel,
            from_=0.0,
            to=250.0,
            resolution=5.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_conductance_change,
            bg="#0D152B",
            troughcolor="#1A243F",
            highlightthickness=0,
        )
        self.slider_gna.set(120.0)
        self.slider_gna.pack(fill=tk.X, pady=(2, 8))

        # g_K
        lbl_gk = tk.Frame(left_panel, bg="#0D152B")
        lbl_gk.pack(fill=tk.X)
        tk.Label(lbl_gk, text="Potassium (g_K):", font=("Helvetica", 9), fg="#94A3B8", bg="#0D152B").pack(side=tk.LEFT)
        self.lbl_gk_val = tk.Label(lbl_gk, text="36 mS", font=("Courier", 9, "bold"), fg="#A855F7", bg="#0D152B")
        self.lbl_gk_val.pack(side=tk.RIGHT)
        self.slider_gk = tk.Scale(
            left_panel,
            from_=0.0,
            to=90.0,
            resolution=2.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_conductance_change,
            bg="#0D152B",
            troughcolor="#1A243F",
            highlightthickness=0,
        )
        self.slider_gk.set(36.0)
        self.slider_gk.pack(fill=tk.X, pady=(2, 8))

        # g_L
        lbl_gl = tk.Frame(left_panel, bg="#0D152B")
        lbl_gl.pack(fill=tk.X)
        tk.Label(lbl_gl, text="Leakage (g_L):", font=("Helvetica", 9), fg="#94A3B8", bg="#0D152B").pack(side=tk.LEFT)
        self.lbl_gl_val = tk.Label(lbl_gl, text="0.30 mS", font=("Courier", 9, "bold"), fg="#10B981", bg="#0D152B")
        self.lbl_gl_val.pack(side=tk.RIGHT)
        self.slider_gl = tk.Scale(
            left_panel,
            from_=0.0,
            to=1.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_conductance_change,
            bg="#0D152B",
            troughcolor="#1A243F",
            highlightthickness=0,
        )
        self.slider_gl.set(0.30)
        self.slider_gl.pack(fill=tk.X, pady=(2, 12))

        # 3. Quick Stimulation Injections
        tk.Label(left_panel, text="SYNAPTIC INJECTION", font=("Helvetica", 10, "bold"), fg="#00F0FF", bg="#0D152B").pack(anchor=tk.W, pady=(8, 6))

        btn_ampa = tk.Button(left_panel, text="Trigger AMPA EPSP", command=self._stim_ampa, bg="#162547", fg="#FFF", relief=tk.FLAT)
        btn_ampa.pack(fill=tk.X, pady=3)

        btn_gaba = tk.Button(left_panel, text="Trigger GABA_A IPSP", command=self._stim_gaba, bg="#162547", fg="#FFF", relief=tk.FLAT)
        btn_gaba.pack(fill=tk.X, pady=3)

        btn_hyper = tk.Button(left_panel, text="Hyperpolarizing Clamp (-6 uA)", command=self._clamp_hyper, bg="#2E1C38", fg="#F43F5E", relief=tk.FLAT)
        btn_hyper.pack(fill=tk.X, pady=3)

        # Right Graphics Display Frame
        right_box = tk.Frame(main_box, bg="#070B16")
        right_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Upper Oscilloscope Canvas (Membrane Potential & Current)
        top_plot_frame = tk.Frame(right_box, bg="#090E1F", highlightbackground="#253556", highlightthickness=1)
        top_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))

        self.canvas_scope = tk.Canvas(top_plot_frame, bg="#050814", highlightthickness=0)
        self.canvas_scope.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas_scope.bind("<Button-1>", self._on_scope_click)

        # Lower Split View: Phase-Plane Plot + Gating / Morphology Plot
        lower_plots = tk.Frame(right_box, bg="#070B16", height=240)
        lower_plots.pack(side=tk.BOTTOM, fill=tk.X)
        lower_plots.pack_propagate(False)

        # Phase-Plane (V vs n or V vs dV/dt)
        phase_frame = tk.Frame(lower_plots, bg="#090E1F", highlightbackground="#253556", highlightthickness=1)
        phase_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.canvas_phase = tk.Canvas(phase_frame, bg="#050814", highlightthickness=0)
        self.canvas_phase.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Gating Kinetics (m, h, n vs time)
        gating_frame = tk.Frame(lower_plots, bg="#090E1F", highlightbackground="#253556", highlightthickness=1)
        gating_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.canvas_gates = tk.Canvas(gating_frame, bg="#050814", highlightthickness=0)
        self.canvas_gates.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bottom Telemetry Ribbon
        bot_bar = tk.Frame(self.root, bg="#0C1326", height=32, padx=14, pady=4, highlightbackground="#253556", highlightthickness=1)
        bot_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.hud_v = tk.Label(bot_bar, text="V: -65.0 mV", font=("Courier", 10, "bold"), fg="#00F0FF", bg="#0C1326")
        self.hud_v.pack(side=tk.LEFT, padx=12)

        self.hud_rate = tk.Label(bot_bar, text="Spike Rate: 0.0 Hz", font=("Courier", 10), fg="#10B981", bg="#0C1326")
        self.hud_rate.pack(side=tk.LEFT, padx=12)

        self.hud_ina = tk.Label(bot_bar, text="I_Na: 0.0 uA", font=("Courier", 10), fg="#94A3B8", bg="#0C1326")
        self.hud_ina.pack(side=tk.LEFT, padx=12)

        self.hud_ik = tk.Label(bot_bar, text="I_K: 0.0 uA", font=("Courier", 10), fg="#94A3B8", bg="#0C1326")
        self.hud_ik.pack(side=tk.LEFT, padx=12)

        self.hud_count = tk.Label(bot_bar, text="Spike Count: 0", font=("Courier", 10), fg="#A855F7", bg="#0C1326")
        self.hud_count.pack(side=tk.RIGHT, padx=12)

    def _on_preset_change(self, event=None):
        name = self.preset_var.get()
        key_map = {
            "Hodgkin-Huxley Giant Squid Axon": "giant_squid",
            "Anode Break Excitation (Rebound)": "anode_break",
            "PING Cortical Gamma Oscillations": "ping_gamma",
            "Locomotor Half-Center CPG": "half_center",
            "Dendritic Cable & Back-Propagation": "dendritic",
            "Thalamocortical Bursting vs Tonic": "thalamic",
        }
        key = key_map.get(name, "giant_squid")
        self.current_preset_key = key
        self.circuit, self.metadata = PRESETS[key]()
        self.primary_id = self.metadata.get("primary_id", 0)

        # Synchronize UI sliders
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            self.slider_inj.set(neuron.i_inj)
            self.slider_gna.set(neuron.g_na_bar)
            self.slider_gk.set(neuron.g_k_bar)
            self.slider_gl.set(neuron.g_l)
            self._update_slider_labels()

        self._reset_buffers()

    def _on_inj_change(self, val):
        inj = float(val)
        self.lbl_inj_val.config(text=f"{inj:.1f} uA/cm2")
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.i_inj = inj
        if self.circuit.multi_compartment:
            self.circuit.multi_compartment.compartments[0].i_inj = inj

    def _on_conductance_change(self, val=None):
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.g_na_bar = self.slider_gna.get()
            neuron.g_k_bar = self.slider_gk.get()
            neuron.g_l = self.slider_gl.get()
        self._update_slider_labels()

    def _update_slider_labels(self):
        self.lbl_inj_val.config(text=f"{self.slider_inj.get():.1f} uA/cm2")
        self.lbl_gna_val.config(text=f"{self.slider_gna.get():.0f} mS")
        self.lbl_gk_val.config(text=f"{self.slider_gk.get():.0f} mS")
        self.lbl_gl_val.config(text=f"{self.slider_gl.get():.2f} mS")

    def _toggle_pause(self):
        self.running = not self.running
        self.btn_pause.config(text="Play" if not self.running else "Pause")

    def _step_once(self):
        self.running = False
        self.btn_pause.config(text="Play")
        self._step_simulation()
        self._draw_displays()

    def _reset_sim(self):
        self.circuit, self.metadata = PRESETS[self.current_preset_key]()
        self.primary_id = self.metadata.get("primary_id", 0)
        self._reset_buffers()
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.i_inj = self.slider_inj.get()
            neuron.g_na_bar = self.slider_gna.get()
            neuron.g_k_bar = self.slider_gk.get()
            neuron.g_l = self.slider_gl.get()

    def _reset_buffers(self):
        self.trace_v.clear()
        self.trace_i.clear()
        self.trace_m.clear()
        self.trace_h.clear()
        self.trace_n.clear()
        self.trace_v2.clear()
        self.spike_count = 0
        self.firing_rate_hz = 0.0

    def _stim_pulse(self):
        """Deliver 2 ms high-intensity current pulse."""
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.i_inj += 20.0
            self.root.after(8, lambda: self._remove_pulse(20.0))

    def _remove_pulse(self, delta):
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.i_inj -= delta

    def _stim_ampa(self):
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.receive_spike("AMPA", weight=2.4)

    def _stim_gaba(self):
        neuron = self.circuit.neurons.get(self.primary_id)
        if neuron:
            neuron.receive_spike("GABA", weight=3.2)

    def _clamp_hyper(self):
        self.slider_inj.set(-6.0)
        self._on_inj_change("-6.0")

    def _on_scope_click(self, event):
        """Click on oscilloscope to trigger patch-clamp current stimulation."""
        self._stim_pulse()

    def _step_simulation(self):
        # Run 6 internal substeps for high fidelity per frame
        substeps = 6
        dt_sub = self.dt
        for _ in range(substeps):
            self.circuit.step(dt_sub)

        neuron = self.circuit.neurons.get(self.primary_id)
        if not neuron:
            return

        # Record oscilloscope history
        self.trace_v.append(neuron.v)
        self.trace_i.append(neuron.i_inj)
        self.trace_m.append(neuron.m)
        self.trace_h.append(neuron.h)
        self.trace_n.append(neuron.n)

        # Record second channel if present (secondary neuron or apical dendrite)
        sec_id = self.metadata.get("secondary_id")
        if sec_id is not None and sec_id in self.circuit.neurons:
            self.trace_v2.append(self.circuit.neurons[sec_id].v)
        elif self.circuit.multi_compartment:
            self.trace_v2.append(self.circuit.multi_compartment.compartments[3].v)
        else:
            self.trace_v2.append(neuron.v)

        if len(self.trace_v) > self.trace_length:
            self.trace_v.pop(0)
            self.trace_i.pop(0)
            self.trace_m.pop(0)
            self.trace_h.pop(0)
            self.trace_n.pop(0)
            self.trace_v2.pop(0)

        # Firing rate calculation over last 500 ms window
        self.spike_count = len(neuron.spike_times)
        cur_t = self.circuit.sim_time
        recent_spikes = [t for t in neuron.spike_times if cur_t - t <= 500.0]
        if len(recent_spikes) >= 2:
            duration_s = (recent_spikes[-1] - recent_spikes[0]) / 1000.0
            if duration_s > 0.005:
                self.firing_rate_hz = (len(recent_spikes) - 1) / duration_s
        else:
            self.firing_rate_hz = 0.0

    def _draw_displays(self):
        neuron = self.circuit.neurons.get(self.primary_id)
        if not neuron:
            return

        # 1. Draw Oscilloscope
        cw = self.canvas_scope.winfo_width()
        ch = self.canvas_scope.winfo_height()
        if cw > 10 and ch > 10:
            self.canvas_scope.delete("all")

            # Grid lines
            v_min, v_max = self.metadata.get("voltage_range", (-90.0, 50.0))
            dv = v_max - v_min

            # Voltage grid lines (-80, -60, -40, -20, 0, +20, +40)
            for v_grid in [-80, -60, -40, -20, 0, 20, 40]:
                if v_min <= v_grid <= v_max:
                    y = ch - ((v_grid - v_min) / dv) * (ch - 40) - 20
                    self.canvas_scope.create_line(45, y, cw - 15, y, fill="#111B35", dash=(2, 4))
                    self.canvas_scope.create_text(25, y, text=f"{v_grid:+d}", fill="#64748B", font=("Courier", 8))

            # Action potential threshold line (-50 mV)
            y_thresh = ch - ((-50.0 - v_min) / dv) * (ch - 40) - 20
            self.canvas_scope.create_line(45, y_thresh, cw - 15, y_thresh, fill="rgba(244,63,94,0.4)", dash=(3, 3))
            self.canvas_scope.create_text(cw - 55, y_thresh - 8, text="Threshold", fill="#F43F5E", font=("Helvetica", 8))

            # Draw Channel 1: Primary Membrane Potential (Phosphor Cyan)
            pts_v = []
            pts_v2 = []
            n_pts = len(self.trace_v)
            if n_pts > 1:
                dx = (cw - 60) / float(self.trace_length)
                for idx in range(n_pts):
                    x = 45 + idx * dx
                    y = ch - ((self.trace_v[idx] - v_min) / dv) * (ch - 40) - 20
                    pts_v.extend([x, y])

                    if len(self.trace_v2) > idx:
                        y2 = ch - ((self.trace_v2[idx] - v_min) / dv) * (ch - 40) - 20
                        pts_v2.extend([x, y2])

                # Secondary Channel (Purple)
                if len(pts_v2) >= 4 and (self.metadata.get("secondary_id") is not None or self.circuit.multi_compartment):
                    self.canvas_scope.create_line(pts_v2, fill="#A855F7", width=1.5, smooth=True)

                # Primary Channel (Cyan)
                self.canvas_scope.create_line(pts_v, fill="#00F0FF", width=2.0, smooth=True)

            # Legends
            self.canvas_scope.create_text(70, 18, text="CH1: V_soma (mV)", fill="#00F0FF", font=("Helvetica", 9, "bold"))
            if self.metadata.get("secondary_id") is not None:
                self.canvas_scope.create_text(200, 18, text="CH2: V_interneuron", fill="#A855F7", font=("Helvetica", 9, "bold"))
            elif self.circuit.multi_compartment:
                self.canvas_scope.create_text(200, 18, text="CH2: V_dendrite_tuft", fill="#A855F7", font=("Helvetica", 9, "bold"))

        # 2. Draw Phase-Plane Trajectory (V vs n)
        pw = self.canvas_phase.winfo_width()
        ph = self.canvas_phase.winfo_height()
        if pw > 10 and ph > 10:
            self.canvas_phase.delete("all")
            self.canvas_phase.create_text(pw // 2, 14, text="PHASE-PLANE: V vs n (Limit Cycle)", fill="#00F0FF", font=("Helvetica", 9, "bold"))

            # Axes
            v_min, v_max = -90.0, 50.0
            n_min, n_max = 0.0, 1.0

            # Plot trajectory
            if len(self.trace_v) > 2 and len(self.trace_n) > 2:
                pts_phase = []
                for idx in range(len(self.trace_v)):
                    px = 35 + ((self.trace_v[idx] - v_min) / (v_max - v_min)) * (pw - 60)
                    py = ph - 25 - ((self.trace_n[idx] - n_min) / (n_max - n_min)) * (ph - 50)
                    pts_phase.extend([px, py])

                if len(pts_phase) >= 4:
                    self.canvas_phase.create_line(pts_phase, fill="#10B981", width=1.5, smooth=True)

                # Current state point
                cur_x = 35 + ((neuron.v - v_min) / (v_max - v_min)) * (pw - 60)
                cur_y = ph - 25 - ((neuron.n - n_min) / (n_max - n_min)) * (ph - 50)
                self.canvas_phase.create_oval(cur_x - 4, cur_y - 4, cur_x + 4, cur_y + 4, fill="#00F0FF", outline="#FFF")

            self.canvas_phase.create_text(pw // 2, ph - 10, text="Membrane Voltage V (mV)", fill="#64748B", font=("Helvetica", 8))
            self.canvas_phase.create_text(18, ph // 2, text="n", fill="#64748B", font=("Helvetica", 8))

        # 3. Draw Gating Variables (m, h, n vs time)
        gw = self.canvas_gates.winfo_width()
        gh = self.canvas_gates.winfo_height()
        if gw > 10 and gh > 10:
            self.canvas_gates.delete("all")
            self.canvas_gates.create_text(gw // 2, 14, text="GATING PARTICLES: m, h, n", fill="#F1F5F9", font=("Helvetica", 9, "bold"))

            if len(self.trace_m) > 1:
                dx = (gw - 50) / float(self.trace_length)
                pts_m, pts_h, pts_n = [], [], []
                for idx in range(len(self.trace_m)):
                    x = 30 + idx * dx
                    ym = gh - 25 - self.trace_m[idx] * (gh - 45)
                    yh = gh - 25 - self.trace_h[idx] * (gh - 45)
                    yn = gh - 25 - self.trace_n[idx] * (gh - 45)
                    pts_m.extend([x, ym])
                    pts_h.extend([x, yh])
                    pts_n.extend([x, yn])

                if len(pts_m) >= 4:
                    self.canvas_gates.create_line(pts_m, fill="#00F0FF", width=1.5)  # m gate (cyan)
                    self.canvas_gates.create_line(pts_h, fill="#F43F5E", width=1.5)  # h gate (rose)
                    self.canvas_gates.create_line(pts_n, fill="#10B981", width=1.5)  # n gate (emerald)

            # Gate labels
            self.canvas_gates.create_text(gw - 110, gh - 10, text="m (Na act)", fill="#00F0FF", font=("Helvetica", 8))
            self.canvas_gates.create_text(gw - 65, gh - 10, text="h (Na inact)", fill="#F43F5E", font=("Helvetica", 8))
            self.canvas_gates.create_text(gw - 20, gh - 10, text="n (K act)", fill="#10B981", font=("Helvetica", 8))

        # 4. Update Bottom Telemetry HUD
        self.hud_v.config(text=f"V: {neuron.v:+.1f} mV")
        self.hud_rate.config(text=f"Spike Rate: {self.firing_rate_hz:.1f} Hz")
        i_na = neuron.g_na_bar * (neuron.m ** 3) * neuron.h * (neuron.v - neuron.e_na)
        i_k = neuron.g_k_bar * (neuron.n ** 4) * (neuron.v - neuron.e_k)
        self.hud_ina.config(text=f"I_Na: {i_na:+.1f} uA")
        self.hud_ik.config(text=f"I_K: {i_k:+.1f} uA")
        self.hud_count.config(text=f"Spike Count: {self.spike_count}")

    def _animate(self):
        if self.running:
            self._step_simulation()
            self._draw_displays()
        self.root.after(25, self._animate)


def main():
    root = tk.Tk()
    app = NeuroSimApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
