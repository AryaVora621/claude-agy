#!/usr/bin/env python3
"""PyCircuit - Standalone Desktop Interactive SPICE Analog Circuit Simulator & Schematic Designer.

Built with Python standard library Tkinter with zero external dependencies.
Features real-time Modified Nodal Analysis (MNA), animated electron flow,
dynamic voltage color-coding, interactive switches, multi-trace oscilloscope dock,
and curated electrical engineering circuit presets.
"""

from __future__ import annotations
import math
import sys
import time
import os
from typing import List, Tuple, Optional, Dict, Any

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: Tkinter is required to run PyCircuit Desktop GUI.")
    sys.exit(1)

try:
    from .engine import (
        Circuit, Resistor, Capacitor, Inductor, VoltageSourceDC,
        VoltageSourceAC, VoltageSourceClock, Diode, Switch, OpAmp, CircuitComponent
    )
    from .presets import PRESETS
except ImportError:
    from engine import (
        Circuit, Resistor, Capacitor, Inductor, VoltageSourceDC,
        VoltageSourceAC, VoltageSourceClock, Diode, Switch, OpAmp, CircuitComponent
    )
    from presets import PRESETS


class PyCircuitApp:
    """Desktop Interactive SPICE Analog Circuit Studio."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("PyCircuit - Interactive SPICE Analog Circuit Simulator")
        self.root.geometry("1360x900")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#0B0F19")

        # SPICE Engine
        self.circuit = Circuit()

        # Simulation Clock & State
        self.running = True
        self.sim_speed = 1.0
        self.steps_per_frame = 20
        self.selected_component: Optional[CircuitComponent] = None

        # Oscilloscope Probes
        self.probe_ch1_node = 1  # Cyan trace
        self.probe_ch2_node = 3  # Amber trace
        self.volts_div = 2.0     # V / division
        self.time_div = 2.0      # ms / division

        # Animation state for electron flow
        self.electron_phase = 0.0
        self._loop_running = True
        self.last_frame_time = time.time()
        self.fps = 60.0

        # Build theme and UI
        self._apply_theme()
        self._build_layout()

        # Load initial preset
        self.load_preset("RLC Resonant Tank")

        # Window protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(16, self._update_loop)

    def _apply_theme(self) -> None:
        """Apply dark palette styling."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#F9FAFB", font=("Helvetica", 10))
        style.configure("Header.TLabel", background="#111827", foreground="#06B6D4", font=("Helvetica", 10, "bold"))
        style.configure("TButton", background="#1F293D", foreground="#F9FAFB", font=("Helvetica", 9, "bold"), padding=5)
        style.map("TButton", background=[("active", "#06B6D4"), ("pressed", "#0891B2")], foreground=[("active", "#0B0F19")])
        style.configure("TCombobox", fieldbackground="#1F293D", background="#1F293D", foreground="#F9FAFB")
        style.configure("Horizontal.TScale", background="#111827", troughcolor="#1F293D")

    def _build_layout(self) -> None:
        """Construct top ribbon, center schematic canvas, right sidebar, and bottom oscilloscope dock."""
        # Top Control Ribbon
        top_bar = tk.Frame(self.root, bg="#111827", height=50, highlightthickness=1, highlightbackground="#1F293D")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # Brand Badge
        brand_frame = tk.Frame(top_bar, bg="#111827")
        brand_frame.pack(side=tk.LEFT, padx=16, pady=6)

        logo_lbl = tk.Label(brand_frame, text="PYCIRCUIT", bg="#10B981", fg="#0B0F19", font=("Helvetica", 10, "bold"), padx=8, pady=2)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        sub_lbl = tk.Label(brand_frame, text="SPICE Analog Circuit Simulator & Schematic Designer", bg="#111827", fg="#9CA3AF", font=("Helvetica", 10))
        sub_lbl.pack(side=tk.LEFT)

        # Presets Dropdown
        preset_frame = tk.Frame(top_bar, bg="#111827")
        preset_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(preset_frame, text="Preset Circuit:", bg="#111827", fg="#9CA3AF", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.cb_preset = ttk.Combobox(preset_frame, values=list(PRESETS.keys()), state="readonly", width=26)
        self.cb_preset.set("RLC Resonant Tank")
        self.cb_preset.pack(side=tk.LEFT)
        self.cb_preset.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.cb_preset.get()))

        # Action Buttons
        act_frame = tk.Frame(top_bar, bg="#111827")
        act_frame.pack(side=tk.RIGHT, padx=16)

        self.btn_run = tk.Button(act_frame, text="❚❚ Pause", bg="#1F293D", fg="#F9FAFB",
                                font=("Helvetica", 9, "bold"), relief="flat", padx=10, pady=4,
                                command=self.toggle_run)
        self.btn_run.pack(side=tk.LEFT, padx=4)

        btn_step = tk.Button(act_frame, text="⏭ Step", bg="#1F293D", fg="#F9FAFB",
                             font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                             command=self.step_once)
        btn_step.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(act_frame, text="↺ Reset", bg="#1F293D", fg="#F9FAFB",
                              font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                              command=self.reset_circuit)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Main Split Workspace
        workspace = tk.Frame(self.root, bg="#080C14")
        workspace.pack(fill=tk.BOTH, expand=True)

        # Right: Parameter & Inspector Sidebar
        sidebar = tk.Frame(workspace, bg="#111827", width=340, highlightthickness=1, highlightbackground="#1F293D")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # Left Column: Schematic Canvas (Top) + Oscilloscope Dock (Bottom)
        left_col = tk.Frame(workspace, bg="#080C14")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Schematic Canvas
        self.canvas_frame = tk.Frame(left_col, bg="#05080E", highlightthickness=1, highlightbackground="#1F293D")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#05080E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Bottom Oscilloscope Dock
        scope_dock = tk.Frame(left_col, bg="#0A0F1D", height=230, highlightthickness=1, highlightbackground="#1F293D")
        scope_dock.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(4, 8))
        scope_dock.pack_propagate(False)

        self._build_oscilloscope_dock(scope_dock)

    def _build_sidebar(self, p: tk.Frame) -> None:
        """Create component inspector, probe selector, and telemetry panel."""
        # --- Section 1: Component Inspector ---
        sec1 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec1.pack(fill=tk.X, padx=12, pady=(12, 6))

        tk.Label(sec1, text="COMPONENT INSPECTOR", bg="#161F30", fg="#10B981", font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.lbl_comp_name = tk.Label(sec1, text="Select a component on canvas", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 9))
        self.lbl_comp_name.pack(anchor="w", pady=(4, 2))

        self.lbl_comp_v = tk.Label(sec1, text="Voltage: -- V", bg="#161F30", fg="#06B6D4", font=("Menlo", 9))
        self.lbl_comp_v.pack(anchor="w")

        self.lbl_comp_i = tk.Label(sec1, text="Current: -- mA", bg="#161F30", fg="#F59E0B", font=("Menlo", 9))
        self.lbl_comp_i.pack(anchor="w")

        # Value adjustment slider
        self.slider_frame = tk.Frame(sec1, bg="#161F30")
        self.slider_frame.pack(fill=tk.X, pady=(8, 0))

        self.lbl_val_title = tk.Label(self.slider_frame, text="Value:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8))
        self.lbl_val_title.pack(anchor="w")

        self.scale_val = ttk.Scale(self.slider_frame, from_=1.0, to=1000.0, orient=tk.HORIZONTAL, command=self._on_slider_value_change)
        self.scale_val.pack(fill=tk.X)

        self.lbl_val_disp = tk.Label(self.slider_frame, text="--", bg="#161F30", fg="#F9FAFB", font=("Menlo", 8, "bold"))
        self.lbl_val_disp.pack(anchor="e")

        # --- Section 2: Oscilloscope Probe Assign ---
        sec2 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec2.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(sec2, text="OSCILLOSCOPE PROBES", bg="#161F30", fg="#06B6D4", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # Channel 1 Probe (Cyan)
        f_ch1 = tk.Frame(sec2, bg="#161F30")
        f_ch1.pack(fill=tk.X, pady=(4, 2))
        tk.Label(f_ch1, text="Probe A (Cyan): Node", bg="#161F30", fg="#06B6D4", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.sp_ch1 = ttk.Spinbox(f_ch1, from_=0, to=10, width=4, command=self._on_probe_change)
        self.sp_ch1.set(self.probe_ch1_node)
        self.sp_ch1.pack(side=tk.RIGHT)

        # Channel 2 Probe (Amber)
        f_ch2 = tk.Frame(sec2, bg="#161F30")
        f_ch2.pack(fill=tk.X, pady=2)
        tk.Label(f_ch2, text="Probe B (Amber): Node", bg="#161F30", fg="#F59E0B", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.sp_ch2 = ttk.Spinbox(f_ch2, from_=0, to=10, width=4, command=self._on_probe_change)
        self.sp_ch2.set(self.probe_ch2_node)
        self.sp_ch2.pack(side=tk.RIGHT)

        # Volts/Div Scale
        f_vdiv = tk.Frame(sec2, bg="#161F30")
        f_vdiv.pack(fill=tk.X, pady=(6, 2))
        tk.Label(f_vdiv, text="Scale (V/div):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(side=tk.LEFT)
        self.scale_vdiv = ttk.Scale(f_vdiv, from_=0.5, to=10.0, value=self.volts_div, orient=tk.HORIZONTAL,
                                    command=lambda v: setattr(self, "volts_div", float(v)))
        self.scale_vdiv.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        # --- Section 3: Simulation Parameters ---
        sec3 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec3.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(sec3, text="MNA SOLVER ENGINE", bg="#161F30", fg="#8B5CF6", font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.lbl_time = tk.Label(sec3, text="Time: 0.000 ms", bg="#161F30", fg="#F9FAFB", font=("Menlo", 9))
        self.lbl_time.pack(anchor="w", pady=1)

        self.lbl_nodes = tk.Label(sec3, text="Nodes: 4 (Ground: Node 0)", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8))
        self.lbl_nodes.pack(anchor="w", pady=1)

        self.lbl_fps = tk.Label(sec3, text="FPS: 60.0", bg="#161F30", fg="#10B981", font=("Menlo", 8))
        self.lbl_fps.pack(anchor="w", pady=1)

    def _build_oscilloscope_dock(self, dock: tk.Frame) -> None:
        """Construct bottom dual-trace oscilloscope viewport."""
        header = tk.Frame(dock, bg="#0D1527", height=26)
        header.pack(fill=tk.X)

        tk.Label(header, text="REAL-TIME OSCILLOSCOPE DOCK", bg="#0D1527", fg="#06B6D4", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=10)
        self.scope_info = tk.Label(header, text="PROBE A: Node 1 (Cyan) • PROBE B: Node 3 (Amber) • TIME: 20 kHz Transient",
                                   bg="#0D1527", fg="#9CA3AF", font=("Menlo", 8))
        self.scope_info.pack(side=tk.RIGHT, padx=10)

        self.scope_canvas = tk.Canvas(dock, bg="#05080E", highlightthickness=0)
        self.scope_canvas.pack(fill=tk.BOTH, expand=True)

    # --- Preset Loading & Actions ---

    def load_preset(self, name: str) -> None:
        """Load a predefined circuit preset."""
        builder = PRESETS.get(name)
        if builder:
            builder(self.circuit)
            self.selected_component = None
            self.sp_ch1.configure(to=max(1, self.circuit.num_nodes - 1))
            self.sp_ch2.configure(to=max(1, self.circuit.num_nodes - 1))
            # Auto-assign reasonable probe nodes
            if self.circuit.num_nodes > 3:
                self.probe_ch1_node = 1
                self.probe_ch2_node = self.circuit.num_nodes - 1
                self.sp_ch1.set(self.probe_ch1_node)
                self.sp_ch2.set(self.probe_ch2_node)
            self.render_all()

    def toggle_run(self) -> None:
        """Toggle simulation running state."""
        self.running = not self.running
        self.btn_run.configure(text="▶ Resume" if not self.running else "❚❚ Pause")

    def step_once(self) -> None:
        """Advance one simulation time step manually."""
        self.circuit.step()
        self.render_all()

    def reset_circuit(self) -> None:
        """Reset time and initial conditions."""
        self.load_preset(self.cb_preset.get())

    def _on_probe_change(self) -> None:
        try:
            self.probe_ch1_node = int(self.sp_ch1.get())
            self.probe_ch2_node = int(self.sp_ch2.get())
        except ValueError:
            pass

    def _on_canvas_click(self, event: Any) -> None:
        """Detect clicked component or toggle switch."""
        cx, cy = event.x, event.y
        clicked = None

        for c in self.circuit.components:
            # Check bounding box
            min_x = min(c.x1, c.x2) - 15
            max_x = max(c.x1, c.x2) + 15
            min_y = min(c.y1, c.y2) - 15
            max_y = max(c.y1, c.y2) + 15

            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                clicked = c
                break

        if clicked:
            self.selected_component = clicked
            if isinstance(clicked, Switch):
                # Interactive click toggles switch
                clicked.closed = not clicked.closed
            self._update_inspector()
            self.render_all()

    def _update_inspector(self) -> None:
        """Update sidebar inspector panel for selected component."""
        c = self.selected_component
        if not c:
            self.lbl_comp_name.configure(text="Select a component on canvas")
            self.lbl_comp_v.configure(text="Voltage: -- V")
            self.lbl_comp_i.configure(text="Current: -- mA")
            return

        name_str = f"{c.name} ({type(c).__name__}) [Nodes {c.n1} -> {c.n2}]"
        self.lbl_comp_name.configure(text=name_str)
        self.lbl_comp_v.configure(text=f"Voltage: {c.voltage_drop:+.3f} V")
        self.lbl_comp_i.configure(text=f"Current: {c.current * 1000.0:+.3f} mA")

        if isinstance(c, (Resistor, VoltageSourceDC, VoltageSourceAC)):
            self.lbl_val_title.configure(text="Value:")
            self.lbl_val_disp.configure(text=f"{c.value:.2f}")
            self.scale_val.configure(from_=1.0, to=max(100.0, c.value * 3.0), value=c.value)

    def _on_slider_value_change(self, val_str: str) -> None:
        val = float(val_str)
        if self.selected_component:
            self.selected_component.value = val
            self.lbl_val_disp.configure(text=f"{val:.2f}")

    # --- Render Pipeline ---

    def _update_loop(self) -> None:
        """Frame animation tick."""
        if not self._loop_running:
            return

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        if self.running:
            for _ in range(self.steps_per_frame):
                self.circuit.step()

            # Advance electron motion phase
            self.electron_phase = (self.electron_phase + 0.12) % (Math_PI2 := 2.0 * math.pi)
            self.render_all()

        self.root.after(16, self._update_loop)

    def render_all(self) -> None:
        """Render schematic canvas and oscilloscope dock."""
        self._render_schematic()
        self._render_oscilloscope()
        self._update_telemetry()

    def _render_schematic(self) -> None:
        """Draw components, wires, voltage coloring, and moving electrons."""
        cv = self.canvas
        cv.delete("all")

        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        # 1. Subtle Schematic Grid Dots
        grid_step = 30
        for gx in range(0, w, grid_step):
            for gy in range(0, h, grid_step):
                cv.create_rectangle(gx, gy, gx + 1, gy + 1, fill="#162032", outline="")

        # 2. Draw Components
        for c in self.circuit.components:
            self._draw_component(cv, c)

        # 3. Draw Op-Amps
        for op in self.circuit.opamps:
            self._draw_opamp(cv, op)

        # 4. Draw Node Pins & Labels
        node_positions: Dict[int, List[Tuple[float, float]]] = {}
        for c in self.circuit.components:
            node_positions.setdefault(c.n1, []).append((c.x1, c.y1))
            node_positions.setdefault(c.n2, []).append((c.x2, c.y2))

        for nid, pts in node_positions.items():
            if not pts:
                continue
            v = self.circuit.node_voltages[nid] if nid < len(self.circuit.node_voltages) else 0.0
            color = self._voltage_to_color(v)

            for px, py in pts:
                cv.create_oval(px - 4, py - 4, px + 4, py + 4, fill=color, outline="#FFF", width=1)
                # Node label
                tag_txt = f"GND" if nid == 0 else f"N{nid} ({v:+.1f}V)"
                cv.create_text(px + 8, py - 8, text=tag_txt, fill="#9CA3AF", font=("Menlo", 8), anchor="w")

    def _draw_component(self, cv: tk.Canvas, c: CircuitComponent) -> None:
        """Draw IEEE electrical schematic symbol with voltage color and current animation."""
        x1, y1 = c.x1, c.y1
        x2, y2 = c.x2, c.y2
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return

        ux, uy = dx / dist, dy / dist
        nx, ny = -uy, ux  # Normal vector

        # Selection highlight
        if c is self.selected_component:
            cv.create_line(x1, y1, x2, y2, fill="#06B6D4", width=8, capstyle=tk.ROUND)

        # Voltage color based on average voltage
        v_avg = (self.circuit.node_voltages[c.n1] + self.circuit.node_voltages[c.n2]) * 0.5
        wire_color = self._voltage_to_color(v_avg)

        # Wire leads: 25% from each end
        lead_len = min(25.0, dist * 0.3)
        lx1 = x1 + ux * lead_len
        ly1 = y1 + uy * lead_len
        lx2 = x2 - ux * lead_len
        ly2 = y2 - uy * lead_len

        cv.create_line(x1, y1, lx1, ly1, fill=wire_color, width=2)
        cv.create_line(lx2, ly2, x2, y2, fill=wire_color, width=2)

        # Draw component body symbol
        if isinstance(c, Resistor):
            # Zig-zag resistor
            num_zigs = 6
            zig_w = 8.0
            body_pts = [(lx1, ly1)]
            seg_len = (dist - 2 * lead_len) / num_zigs
            for i in range(num_zigs):
                frac = (i + 0.5) / num_zigs
                px = lx1 + ux * (i + 0.5) * seg_len
                py = ly1 + uy * (i + 0.5) * seg_len
                side = 1.0 if (i % 2 == 0) else -1.0
                body_pts.append((px + nx * zig_w * side, py + ny * zig_w * side))
            body_pts.append((lx2, ly2))
            flat_pts = [coord for pt in body_pts for coord in pt]
            cv.create_line(*flat_pts, fill="#FBBF24", width=2)
            cv.create_text((x1 + x2) * 0.5 + nx * 16, (y1 + y2) * 0.5 + ny * 16, text=f"{c.name}\n{c.value:.0f}Ω", fill="#FBBF24", font=("Menlo", 8))

        elif isinstance(c, Capacitor):
            # Parallel plates
            plate_w = 12.0
            cv.create_line(lx1 + nx * plate_w, ly1 + ny * plate_w, lx1 - nx * plate_w, ly1 - ny * plate_w, fill="#38BDF8", width=3)
            cv.create_line(lx2 + nx * plate_w, ly2 + ny * plate_w, lx2 - nx * plate_w, ly2 - ny * plate_w, fill="#38BDF8", width=3)
            cv.create_text((x1 + x2) * 0.5 + nx * 16, (y1 + y2) * 0.5 + ny * 16, text=f"{c.name}\n{c.value*1e6:.1f}µF", fill="#38BDF8", font=("Menlo", 8))

        elif isinstance(c, Inductor):
            # Coiled loops
            cv.create_line(lx1, ly1, lx2, ly2, fill="#F43F5E", width=3, dash=(4, 2))
            cv.create_text((x1 + x2) * 0.5 + nx * 16, (y1 + y2) * 0.5 + ny * 16, text=f"{c.name}\n{c.value*1000:.0f}mH", fill="#F43F5E", font=("Menlo", 8))

        elif isinstance(c, (VoltageSourceDC, VoltageSourceAC, VoltageSourceClock)):
            # Circle with symbol
            mid_x = (x1 + x2) * 0.5
            mid_y = (y1 + y2) * 0.5
            rad = 14.0
            cv.create_oval(mid_x - rad, mid_y - rad, mid_x + rad, mid_y + rad, fill="#0B132B", outline="#10B981", width=2)
            if isinstance(c, VoltageSourceDC):
                cv.create_text(mid_x, mid_y - 4, text="+", fill="#10B981", font=("Helvetica", 9, "bold"))
                cv.create_text(mid_x, mid_y + 4, text="-", fill="#10B981", font=("Helvetica", 9, "bold"))
                cv.create_text(mid_x + nx * 20, mid_y + ny * 20, text=f"{c.value:.1f}V", fill="#10B981", font=("Menlo", 8))
            else:
                cv.create_text(mid_x, mid_y, text="~", fill="#10B981", font=("Helvetica", 14, "bold"))
                cv.create_text(mid_x + nx * 20, mid_y + ny * 20, text=f"{c.value:.1f}V\n{getattr(c, 'frequency', 60):.0f}Hz", fill="#10B981", font=("Menlo", 8))

        elif isinstance(c, Diode):
            # Triangle + cathode bar
            mid_x = (x1 + x2) * 0.5
            mid_y = (y1 + y2) * 0.5
            tri_w = 8.0
            tri_pts = [
                lx1, ly1,
                mid_x + nx * tri_w, mid_y + ny * tri_w,
                mid_x - nx * tri_w, mid_y - ny * tri_w
            ]
            cv.create_polygon(*tri_pts, fill="#F59E0B" if c.is_conducting else "#374151", outline="#FFF")
            cv.create_line(mid_x + nx * tri_w, mid_y + ny * tri_w, mid_x - nx * tri_w, mid_y - ny * tri_w, fill="#FFF", width=2)
            cv.create_line(mid_x, mid_y, lx2, ly2, fill=wire_color, width=2)

        elif isinstance(c, Switch):
            # Lever switch
            mid_x = (x1 + x2) * 0.5
            mid_y = (y1 + y2) * 0.5
            cv.create_oval(lx1 - 3, ly1 - 3, lx1 + 3, ly1 + 3, fill="#FFF")
            cv.create_oval(lx2 - 3, ly2 - 3, lx2 + 3, ly2 + 3, fill="#FFF")
            lever_end_x = lx2 if c.closed else (lx1 + ux * 18 + nx * 14)
            lever_end_y = ly2 if c.closed else (ly1 + uy * 18 + ny * 14)
            cv.create_line(lx1, ly1, lever_end_x, lever_end_y, fill="#10B981" if c.closed else "#EF4444", width=3)
            cv.create_text(mid_x + nx * 16, mid_y + ny * 16, text=f"[Switch {'ON' if c.closed else 'OFF'}]", fill="#9CA3AF", font=("Menlo", 7))

        # 4. Animated Flowing Electrons along component
        if abs(c.current) > 1e-6:
            dir_sign = 1.0 if c.current > 0 else -1.0
            dot_count = max(2, int(dist / 22.0))
            for k in range(dot_count):
                frac = (self.electron_phase * dir_sign + k / float(dot_count)) % 1.0
                ex = x1 + ux * dist * frac
                ey = y1 + uy * dist * frac
                cv.create_oval(ex - 2, ey - 2, ex + 2, ey + 2, fill="#FDE047", outline="")

    def _draw_opamp(self, cv: tk.Canvas, op: OpAmp) -> None:
        """Render IEEE standard operational amplifier symbol."""
        # Standard triangle: inputs on left (Node 260, 140 / 260, 180), output on right (400, 160)
        tri_pts = [260, 120, 260, 200, 360, 160]
        cv.create_polygon(*tri_pts, fill="#1E293B", outline="#8B5CF6", width=2)
        cv.create_text(272, 140, text="-", fill="#FFF", font=("Helvetica", 11, "bold"))
        cv.create_text(272, 180, text="+", fill="#FFF", font=("Helvetica", 11, "bold"))
        cv.create_text(300, 160, text=op.name, fill="#8B5CF6", font=("Menlo", 9, "bold"))
        # Lead lines
        cv.create_line(360, 160, 460, 160, fill="#8B5CF6", width=2)

    def _voltage_to_color(self, v: float) -> str:
        """Color voltage potential: Green (+) to Cyan (0V) to Red (-)."""
        if v > 0.1:
            intensity = min(1.0, v / 12.0)
            g = int(185 + 70 * intensity)
            return f"#{int(16 + 20 * intensity):02x}{g:02x}{int(129 - 40 * intensity):02x}"
        elif v < -0.1:
            intensity = min(1.0, abs(v) / 12.0)
            r = int(220 + 35 * intensity)
            return f"#{r:02x}{int(60 - 30 * intensity):02x}{int(60 - 30 * intensity):02x}"
        else:
            return "#06B6D4"  # Ground / Neutral potential

    def _render_oscilloscope(self) -> None:
        """Render scrolling waveforms in oscilloscope dock."""
        cv = self.scope_canvas
        cv.delete("all")

        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        mid_y = h / 2.0

        # CRT Graticule
        grid_x = 40
        grid_y = 25
        for x in range(0, w, grid_x):
            cv.create_line(x, 0, x, h, fill="#0E1B2C", width=1, dash=(2, 2))
        for y in range(0, h, grid_y):
            cv.create_line(0, y, w, y, fill="#0E1B2C", width=1, dash=(2, 2))

        # Center Zero Axis
        cv.create_line(0, mid_y, w, mid_y, fill="#1F3552", width=1)

        hist_n1 = self.circuit.history_nodes.get(self.probe_ch1_node, [])
        hist_n2 = self.circuit.history_nodes.get(self.probe_ch2_node, [])

        pixels_per_volt = (grid_y * 2) / max(0.1, self.volts_div)

        # Plot Channel 2 (Amber trace)
        if len(hist_n2) > 1:
            pts2 = []
            step = w / float(len(hist_n2) - 1)
            for i, v in enumerate(hist_n2):
                x = i * step
                y = mid_y - v * pixels_per_volt
                pts2.extend([x, y])
            cv.create_line(*pts2, fill="#F59E0B", width=2)

        # Plot Channel 1 (Cyan trace)
        if len(hist_n1) > 1:
            pts1 = []
            step = w / float(len(hist_n1) - 1)
            for i, v in enumerate(hist_n1):
                x = i * step
                y = mid_y - v * pixels_per_volt
                pts1.extend([x, y])
            cv.create_line(*pts1, fill="#06B6D4", width=2)

        # Telemetry Labels
        v1 = self.circuit.node_voltages[self.probe_ch1_node] if self.probe_ch1_node < len(self.circuit.node_voltages) else 0.0
        v2 = self.circuit.node_voltages[self.probe_ch2_node] if self.probe_ch2_node < len(self.circuit.node_voltages) else 0.0
        cv.create_text(16, 14, text=f"CH1 (Node {self.probe_ch1_node}): {v1:+.2f} V", fill="#06B6D4", font=("Menlo", 9, "bold"), anchor="w")
        cv.create_text(220, 14, text=f"CH2 (Node {self.probe_ch2_node}): {v2:+.2f} V", fill="#F59E0B", font=("Menlo", 9, "bold"), anchor="w")
        cv.create_text(w - 16, 14, text=f"{self.volts_div:.1f} V/div • MNA Transient", fill="#9CA3AF", font=("Menlo", 8), anchor="e")

    def _update_telemetry(self) -> None:
        """Update time and solver status readouts."""
        t_ms = self.circuit.time * 1000.0
        self.lbl_time.configure(text=f"Time: {t_ms:.2f} ms")
        self.lbl_nodes.configure(text=f"Nodes: {self.circuit.num_nodes} (GND: Node 0)")
        self.lbl_fps.configure(text=f"FPS: {self.fps:.1f}")

    def on_close(self) -> None:
        """Clean shutdown handler."""
        self._loop_running = False
        self.root.destroy()


def main() -> None:
    """Application entry point."""
    root = tk.Tk()
    app = PyCircuitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
