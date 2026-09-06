"""
OptiFlow 2D: Standalone Desktop Computational Fluid Dynamics & Aerodynamics Studio
Zero external dependencies. Pure Python standard library Tkinter.

Features:
- Real-time 2D Navier-Stokes solver (Vorticity-Streamfunction with Poisson pressure)
- Interactive NACA 4-digit airfoil generator with live Angle of Attack control
- Multiple flow visualizations: Velocity magnitude, Vorticity, Pressure, Streamlines, Smoke
- Virtual Pitot tube probe and aerodynamic coefficients (CL, CD, L/D, CM)
- Zero external dependencies: standard library Python only.
"""

import sys
import os
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple

# Enable relative imports when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.optiflow.cfd_engine import CFDEngine2D
from programs.optiflow.presets import AERO_PRESETS


class OptiFlowApp:
    """Desktop GUI Application for OptiFlow 2D Aerodynamics Laboratory."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OptiFlow 2D: Computational Fluid Dynamics & Aerodynamics Studio")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)
        self.root.configure(bg="#0B0F19")

        # Computational Engine (80x40 grid for high interactivity)
        self.engine = CFDEngine2D(nx=80, ny=40, lx=2.4, ly=1.2, u_inf=1.0, reynolds=350.0)

        # Simulation State
        self.is_running = True
        self.vis_mode = "velocity" # "velocity", "vorticity", "pressure", "streamfunction"
        self.show_smoke = True
        self.show_vectors = False
        self.show_grid = False
        self.probe_pos: Optional[Tuple[float, float]] = (0.6, 0.6)
        self.current_preset = "naca0012"
        self.alpha_deg = 6.0
        self.camber_m = 0.0
        self.camber_p = 0.0
        self.thick_t = 0.12
        self.brush_mode = "probe" # "probe", "draw_solid", "erase_solid"

        # Apply default preset
        self.engine.set_naca_airfoil(m=0.0, p=0.0, t=0.12, chord=0.6,
                                     x_center=0.75, y_center=0.6, alpha_deg=self.alpha_deg)

        self._create_ui()
        self._animate()

    def _create_ui(self):
        # 1. Top Header & Toolbar
        header_frame = tk.Frame(self.root, bg="#111827", height=48, bd=0, highlightthickness=1,
                                highlightbackground="#1E293B")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        brand_lbl = tk.Label(header_frame, text="OPTIFLOW 2D", fg="#00F0FF", bg="#111827",
                             font=("Helvetica", 13, "bold"))
        brand_lbl.pack(side=tk.LEFT, padx=(16, 8), pady=8)

        sub_lbl = tk.Label(header_frame, text="|  Navier-Stokes Aerodynamics Laboratory",
                           fg="#64748B", bg="#111827", font=("Helvetica", 10))
        sub_lbl.pack(side=tk.LEFT, padx=(0, 20), pady=8)

        # Control buttons
        self.btn_play = tk.Button(header_frame, text="Pause", bg="#F43F5E", fg="#FFFFFF",
                                  activebackground="#E11D48", activeforeground="#FFFFFF",
                                  font=("Helvetica", 9, "bold"), relief=tk.FLAT, padx=12, pady=4,
                                  command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(header_frame, text="Reset Flow", bg="#1E293B", fg="#F1F5F9",
                              activebackground="#334155", activeforeground="#00F0FF",
                              font=("Helvetica", 9), relief=tk.FLAT, padx=10, pady=4,
                              command=self._reset_flow)
        btn_reset.pack(side=tk.LEFT, padx=4)

        btn_clear = tk.Button(header_frame, text="Clear Obstacles", bg="#1E293B", fg="#F1F5F9",
                              activebackground="#334155", activeforeground="#00F0FF",
                              font=("Helvetica", 9), relief=tk.FLAT, padx=10, pady=4,
                              command=self._clear_obstacles)
        btn_clear.pack(side=tk.LEFT, padx=4)

        # Presets selector
        preset_lbl = tk.Label(header_frame, text="Preset:", fg="#94A3B8", bg="#111827",
                              font=("Helvetica", 9, "bold"))
        preset_lbl.pack(side=tk.LEFT, padx=(16, 4))

        self.preset_var = tk.StringVar(value=self.current_preset)
        preset_menu = ttk.Combobox(header_frame, textvariable=self.preset_var, width=24, state="readonly")
        preset_menu["values"] = list(AERO_PRESETS.keys())
        preset_menu.bind("<<ComboboxSelected>>", self._on_preset_change)
        preset_menu.pack(side=tk.LEFT, padx=4)

        # Vis Mode
        vis_lbl = tk.Label(header_frame, text="Display:", fg="#94A3B8", bg="#111827",
                           font=("Helvetica", 9, "bold"))
        vis_lbl.pack(side=tk.LEFT, padx=(16, 4))

        self.vis_var = tk.StringVar(value=self.vis_mode)
        vis_menu = ttk.Combobox(header_frame, textvariable=self.vis_var, width=18, state="readonly")
        vis_menu["values"] = ["velocity", "vorticity", "pressure", "streamfunction"]
        vis_menu.bind("<<ComboboxSelected>>", self._on_vis_change)
        vis_menu.pack(side=tk.LEFT, padx=4)

        # 2. Main Workspace Layout
        main_box = tk.Frame(self.root, bg="#0B0F19")
        main_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Sidebar (Parameters & Telemetry)
        sidebar = tk.Frame(main_box, bg="#111827", width=310, bd=0, highlightthickness=1,
                           highlightbackground="#1E293B")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # Center Fluid Canvas
        center_frame = tk.Frame(main_box, bg="#050811")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center_frame, bg="#050811", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

        # Bottom Telemetry Ribbon
        bottom_ribbon = tk.Frame(self.root, bg="#0F172A", height=38, bd=0, highlightthickness=1,
                                 highlightbackground="#1E293B")
        bottom_ribbon.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_telem_cl = tk.Label(bottom_ribbon, text="CL: +0.48", fg="#10B981", bg="#0F172A",
                                     font=("Helvetica", 10, "bold"))
        self.lbl_telem_cl.pack(side=tk.LEFT, padx=(20, 16), pady=6)

        self.lbl_telem_cd = tk.Label(bottom_ribbon, text="CD: 0.042", fg="#F59E0B", bg="#0F172A",
                                     font=("Helvetica", 10, "bold"))
        self.lbl_telem_cd.pack(side=tk.LEFT, padx=16, pady=6)

        self.lbl_telem_ld = tk.Label(bottom_ribbon, text="L/D: 11.4", fg="#00F0FF", bg="#0F172A",
                                     font=("Helvetica", 10, "bold"))
        self.lbl_telem_ld.pack(side=tk.LEFT, padx=16, pady=6)

        self.lbl_telem_cm = tk.Label(bottom_ribbon, text="CM: -0.012", fg="#A855F7", bg="#0F172A",
                                     font=("Helvetica", 10, "bold"))
        self.lbl_telem_cm.pack(side=tk.LEFT, padx=16, pady=6)

        self.lbl_telem_stall = tk.Label(bottom_ribbon, text="STATUS: LAMINAR ATTACHED", fg="#10B981",
                                        bg="#0F172A", font=("Helvetica", 9, "bold"))
        self.lbl_telem_stall.pack(side=tk.RIGHT, padx=24, pady=6)

    def _build_sidebar(self, parent: tk.Frame):
        # Section Title: Fluid Properties
        lbl_sec1 = tk.Label(parent, text="FLUID DYNAMICS PARAMETERS", fg="#00F0FF", bg="#111827",
                            font=("Helvetica", 9, "bold"))
        lbl_sec1.pack(anchor=tk.W, padx=14, pady=(12, 6))

        # Reynolds Number Slider
        f_re = tk.Frame(parent, bg="#111827")
        f_re.pack(fill=tk.X, padx=14, pady=3)
        self.lbl_re = tk.Label(f_re, text="Reynolds Number Re: 350", fg="#94A3B8", bg="#111827", font=("Helvetica", 9))
        self.lbl_re.pack(anchor=tk.W)
        self.scale_re = tk.Scale(f_re, from_=50, to=2000, resolution=25, orient=tk.HORIZONTAL,
                                 showvalue=False, bg="#1E293B", fg="#00F0FF", highlightthickness=0,
                                 troughcolor="#0B0F19", command=self._on_re_change)
        self.scale_re.set(350)
        self.scale_re.pack(fill=tk.X, pady=(2, 6))

        # Inflow Velocity Slider
        f_u = tk.Frame(parent, bg="#111827")
        f_u.pack(fill=tk.X, padx=14, pady=3)
        self.lbl_u = tk.Label(f_u, text="Freestream Velocity U∞: 1.0 m/s", fg="#94A3B8", bg="#111827", font=("Helvetica", 9))
        self.lbl_u.pack(anchor=tk.W)
        self.scale_u = tk.Scale(f_u, from_=0.2, to=2.5, resolution=0.1, orient=tk.HORIZONTAL,
                                showvalue=False, bg="#1E293B", fg="#00F0FF", highlightthickness=0,
                                troughcolor="#0B0F19", command=self._on_u_change)
        self.scale_u.set(1.0)
        self.scale_u.pack(fill=tk.X, pady=(2, 6))

        # Section Title: Airfoil Geometry
        lbl_sec2 = tk.Label(parent, text="AIRFOIL MORPHOLOGY & ATTACK", fg="#00F0FF", bg="#111827",
                            font=("Helvetica", 9, "bold"))
        lbl_sec2.pack(anchor=tk.W, padx=14, pady=(10, 6))

        # Angle of Attack Slider
        f_aoa = tk.Frame(parent, bg="#111827")
        f_aoa.pack(fill=tk.X, padx=14, pady=3)
        self.lbl_aoa = tk.Label(f_aoa, text=f"Angle of Attack α: {self.alpha_deg:.1f}°", fg="#94A3B8",
                                bg="#111827", font=("Helvetica", 9))
        self.lbl_aoa.pack(anchor=tk.W)
        self.scale_aoa = tk.Scale(f_aoa, from_=-18.0, to=22.0, resolution=1.0, orient=tk.HORIZONTAL,
                                  showvalue=False, bg="#1E293B", fg="#00F0FF", highlightthickness=0,
                                  troughcolor="#0B0F19", command=self._on_aoa_change)
        self.scale_aoa.set(self.alpha_deg)
        self.scale_aoa.pack(fill=tk.X, pady=(2, 6))

        # Camber Slider
        f_m = tk.Frame(parent, bg="#111827")
        f_m.pack(fill=tk.X, padx=14, pady=3)
        self.lbl_m = tk.Label(f_m, text="Max Camber m: 0%", fg="#94A3B8", bg="#111827", font=("Helvetica", 9))
        self.lbl_m.pack(anchor=tk.W)
        self.scale_m = tk.Scale(f_m, from_=0.0, to=0.08, resolution=0.01, orient=tk.HORIZONTAL,
                                showvalue=False, bg="#1E293B", fg="#00F0FF", highlightthickness=0,
                                troughcolor="#0B0F19", command=self._on_camber_change)
        self.scale_m.set(0.0)
        self.scale_m.pack(fill=tk.X, pady=(2, 6))

        # Section Title: Visual Overlays & Tools
        lbl_sec3 = tk.Label(parent, text="VISUAL OVERLAYS & TOOL", fg="#00F0FF", bg="#111827",
                            font=("Helvetica", 9, "bold"))
        lbl_sec3.pack(anchor=tk.W, padx=14, pady=(10, 4))

        self.chk_smoke_var = tk.BooleanVar(value=True)
        chk_smoke = tk.Checkbutton(parent, text="Smoke Streaklines (Tracer Rake)", variable=self.chk_smoke_var,
                                   bg="#111827", fg="#F1F5F9", selectcolor="#1E293B",
                                   activebackground="#111827", activeforeground="#00F0FF",
                                   command=self._on_smoke_toggle)
        chk_smoke.pack(anchor=tk.W, padx=14, pady=2)

        self.chk_vec_var = tk.BooleanVar(value=False)
        chk_vec = tk.Checkbutton(parent, text="Velocity Vector Quiver Arrows", variable=self.chk_vec_var,
                                 bg="#111827", fg="#F1F5F9", selectcolor="#1E293B",
                                 activebackground="#111827", activeforeground="#00F0FF",
                                 command=self._on_vec_toggle)
        chk_vec.pack(anchor=tk.W, padx=14, pady=2)

        # Tool Radio Buttons
        f_tool = tk.Frame(parent, bg="#111827")
        f_tool.pack(fill=tk.X, padx=14, pady=4)
        self.tool_var = tk.StringVar(value="probe")
        r1 = tk.Radiobutton(f_tool, text="Pitot Probe", value="probe", variable=self.tool_var,
                            bg="#111827", fg="#F1F5F9", selectcolor="#1E293B",
                            activebackground="#111827", command=self._on_tool_change)
        r1.pack(side=tk.LEFT)
        r2 = tk.Radiobutton(f_tool, text="Draw Solid", value="draw_solid", variable=self.tool_var,
                            bg="#111827", fg="#F1F5F9", selectcolor="#1E293B",
                            activebackground="#111827", command=self._on_tool_change)
        r2.pack(side=tk.LEFT, padx=6)
        r3 = tk.Radiobutton(f_tool, text="Erase", value="erase_solid", variable=self.tool_var,
                            bg="#111827", fg="#F1F5F9", selectcolor="#1E293B",
                            activebackground="#111827", command=self._on_tool_change)
        r3.pack(side=tk.LEFT)

        # Virtual Pitot Probe Telemetry Box
        lbl_sec4 = tk.Label(parent, text="VIRTUAL PITOT PROBE READOUT", fg="#00F0FF", bg="#111827",
                            font=("Helvetica", 9, "bold"))
        lbl_sec4.pack(anchor=tk.W, padx=14, pady=(10, 4))

        self.probe_frame = tk.Frame(parent, bg="#0F172A", bd=0, highlightthickness=1,
                                    highlightbackground="#1E293B")
        self.probe_frame.pack(fill=tk.X, padx=14, pady=4)

        self.lbl_probe_coords = tk.Label(self.probe_frame, text="Probe (x, y): (0.60 m, 0.60 m)",
                                         fg="#94A3B8", bg="#0F172A", font=("Courier", 9))
        self.lbl_probe_coords.pack(anchor=tk.W, padx=8, pady=(4, 2))

        self.lbl_probe_vel = tk.Label(self.probe_frame, text="|V|: 1.05 m/s (u: 1.02, v: 0.15)",
                                      fg="#00F0FF", bg="#0F172A", font=("Courier", 9, "bold"))
        self.lbl_probe_vel.pack(anchor=tk.W, padx=8, pady=2)

        self.lbl_probe_press = tk.Label(self.probe_frame, text="Static Pressure p: -12.4 Pa",
                                        fg="#F59E0B", bg="#0F172A", font=("Courier", 9))
        self.lbl_probe_press.pack(anchor=tk.W, padx=8, pady=2)

        self.lbl_probe_cp = tk.Label(self.probe_frame, text="Cp: -0.21 | Dynamic q: 0.67 Pa",
                                     fg="#A855F7", bg="#0F172A", font=("Courier", 9))
        self.lbl_probe_cp.pack(anchor=tk.W, padx=8, pady=(2, 6))

    def _toggle_play(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_play.config(text="Pause", bg="#F43F5E")
        else:
            self.btn_play.config(text="Resume", bg="#10B981")

    def _reset_flow(self):
        self.engine.reset_flow()

    def _clear_obstacles(self):
        self.engine.clear_geometry()

    def _on_re_change(self, val):
        re = float(val)
        self.lbl_re.config(text=f"Reynolds Number Re: {int(re)}")
        self.engine.reynolds = re
        self.engine.nu = (self.engine.u_inf * self.engine.ly) / re

    def _on_u_change(self, val):
        u = float(val)
        self.lbl_u.config(text=f"Freestream Velocity U∞: {u:.1f} m/s")
        self.engine.u_inf = u
        self.engine.nu = (u * self.engine.ly) / self.engine.reynolds

    def _on_aoa_change(self, val):
        self.alpha_deg = float(val)
        self.lbl_aoa.config(text=f"Angle of Attack α: {self.alpha_deg:.1f}°")
        self.engine.set_naca_airfoil(m=self.camber_m, p=self.camber_p, t=self.thick_t,
                                     chord=0.6, x_center=0.75, y_center=0.6,
                                     alpha_deg=self.alpha_deg)

    def _on_camber_change(self, val):
        self.camber_m = float(val)
        self.camber_p = 0.4 if self.camber_m > 0 else 0.0
        self.lbl_m.config(text=f"Max Camber m: {int(self.camber_m * 100)}%")
        self.engine.set_naca_airfoil(m=self.camber_m, p=self.camber_p, t=self.thick_t,
                                     chord=0.6, x_center=0.75, y_center=0.6,
                                     alpha_deg=self.alpha_deg)

    def _on_smoke_toggle(self):
        self.show_smoke = self.chk_smoke_var.get()

    def _on_vec_toggle(self):
        self.show_vectors = self.chk_vec_var.get()

    def _on_tool_change(self):
        self.brush_mode = self.tool_var.get()

    def _on_vis_change(self, event):
        self.vis_mode = self.vis_var.get()

    def _on_preset_change(self, event):
        key = self.preset_var.get()
        if key not in AERO_PRESETS:
            return
        p = AERO_PRESETS[key]
        ptype = p["type"]
        self.engine.reset_flow()

        if ptype == "naca":
            self.camber_m = p.get("m", 0.0)
            self.camber_p = p.get("p", 0.0)
            self.thick_t = p.get("t", 0.12)
            self.alpha_deg = p.get("alpha", 5.0)
            self.scale_aoa.set(self.alpha_deg)
            self.scale_m.set(self.camber_m)
            self.engine.set_naca_airfoil(m=self.camber_m, p=self.camber_p, t=self.thick_t,
                                         chord=p.get("chord", 0.55), x_center=0.75,
                                         y_center=0.6, alpha_deg=self.alpha_deg)
        elif ptype == "cylinder":
            self.engine.set_circular_cylinder(radius=p.get("radius", 0.08), x_center=0.75, y_center=0.6)
        elif ptype == "venturi":
            self.engine.set_venturi_nozzle(throat_height=p.get("throat", 0.35), inlet_height=p.get("inlet", 0.75))
        elif ptype == "step":
            self.engine.set_backward_facing_step(step_x=p.get("step_x", 0.5), step_h=p.get("step_h", 0.35))

        if "reynolds" in p:
            self.scale_re.set(p["reynolds"])
            self._on_re_change(p["reynolds"])

    def _on_canvas_click(self, event):
        self._handle_canvas_action(event.x, event.y)

    def _on_canvas_drag(self, event):
        self._handle_canvas_action(event.x, event.y)

    def _handle_canvas_action(self, sx: int, sy: int):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 0 or ch <= 0:
            return

        wx = (sx / cw) * self.engine.lx
        wy = (1.0 - sy / ch) * self.engine.ly

        if self.brush_mode == "probe":
            self.probe_pos = (wx, wy)
            self._update_probe_readout()
        elif self.brush_mode == "draw_solid":
            gx = int(wx / self.engine.dx)
            gy = int(wy / self.engine.dy)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    nx = gx + di; ny = gy + dj
                    if 0 <= nx < self.engine.nx and 0 <= ny < self.engine.ny:
                        self.engine.solid[ny][nx] = True
            self.engine.update_boundary_normals()
        elif self.brush_mode == "erase_solid":
            gx = int(wx / self.engine.dx)
            gy = int(wy / self.engine.dy)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    nx = gx + di; ny = gy + dj
                    if 0 <= nx < self.engine.nx and 0 <= ny < self.engine.ny:
                        self.engine.solid[ny][nx] = False
            self.engine.update_boundary_normals()

    def _update_probe_readout(self):
        if not self.probe_pos:
            return
        px, py = self.probe_pos
        telem = self.engine.get_probe_telemetry(px, py)

        self.lbl_probe_coords.config(text=f"Probe (x, y): ({px:.2f} m, {py:.2f} m)")
        if telem["is_solid"] > 0.5:
            self.lbl_probe_vel.config(text="SOLID OBSTACLE BOUNDARY", fg="#F43F5E")
            self.lbl_probe_press.config(text=f"Wall Pressure p: {telem['pressure']:.1f} Pa")
            self.lbl_probe_cp.config(text="Cp: 1.00 (Stagnation)", fg="#94A3B8")
        else:
            self.lbl_probe_vel.config(text=f"|V|: {telem['velocity']:.2f} m/s (u:{telem['u']:.2f}, v:{telem['v']:.2f})",
                                      fg="#00F0FF")
            self.lbl_probe_press.config(text=f"Static Pressure p: {telem['pressure']:.1f} Pa")
            self.lbl_probe_cp.config(text=f"Cp: {telem['cp']:.2f} | Dyn q: {telem['dynamic_pressure']:.2f} Pa",
                                     fg="#A855F7")

    def _get_palette_color(self, val_norm: float, mode: str) -> str:
        """Map normalized value in [0, 1] to hexadecimal RGB color."""
        clamped = max(0.0, min(1.0, val_norm))

        if mode == "velocity":
            # Deep Navy -> Cyan -> Emerald -> Yellow -> White
            if clamped < 0.25:
                t = clamped / 0.25
                r = int(5 + t * (0 - 5))
                g = int(8 + t * (200 - 8))
                b = int(25 + t * (255 - 25))
            elif clamped < 0.50:
                t = (clamped - 0.25) / 0.25
                r = int(0 + t * (16 - 0))
                g = int(200 + t * (240 - 200))
                b = int(255 + t * (120 - 255))
            elif clamped < 0.75:
                t = (clamped - 0.50) / 0.25
                r = int(16 + t * (245 - 16))
                g = int(240 + t * (180 - 240))
                b = int(120 + t * (20 - 120))
            else:
                t = (clamped - 0.75) / 0.25
                r = int(245 + t * 10)
                g = int(180 + t * 75)
                b = int(20 + t * 235)
            return f"#{r:02x}{g:02x}{b:02x}"

        elif mode == "vorticity":
            # Bipolar: Blue (negative clockwise) -> Dark Navy -> Red (positive counter-clockwise)
            if clamped < 0.5:
                t = (0.5 - clamped) / 0.5
                r = int(10 + t * 10)
                g = int(20 + t * 120)
                b = int(50 + t * 205)
            else:
                t = (clamped - 0.5) / 0.5
                r = int(20 + t * 235)
                g = int(20 + t * 40)
                b = int(40 + t * 20)
            return f"#{r:02x}{g:02x}{b:02x}"

        elif mode == "pressure":
            # Jet / Turbo: Blue -> Cyan -> Yellow -> Red
            if clamped < 0.33:
                t = clamped / 0.33
                r = int(10 + t * 0)
                g = int(20 + t * 220)
                b = int(180 + t * 75)
            elif clamped < 0.66:
                t = (clamped - 0.33) / 0.33
                r = int(0 + t * 240)
                g = int(220 + t * 20)
                b = int(255 - t * 240)
            else:
                t = (clamped - 0.66) / 0.34
                r = int(240 + t * 15)
                g = int(240 - t * 200)
                b = int(15 - t * 10)
            return f"#{r:02x}{g:02x}{b:02x}"

        else: # Streamfunction
            r = int(20 + clamped * 180)
            g = int(40 + clamped * 120)
            b = int(100 + clamped * 155)
            return f"#{r:02x}{g:02x}{b:02x}"

    def _render_canvas(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        nx = self.engine.nx
        ny = self.engine.ny
        dx_px = cw / (nx - 1)
        dy_px = ch / (ny - 1)

        # 1. Background Contours
        max_u = max(0.01, self.engine.max_velocity)

        # Draw discrete contour block tiles
        # Group tiles to maintain 60 FPS performance without lagging Tkinter
        stride_x = 2
        stride_y = 2
        w_tile = dx_px * stride_x + 1
        h_tile = dy_px * stride_y + 1

        for j in range(0, ny - 1, stride_y):
            sy = ch - (j + stride_y) * dy_px
            for i in range(0, nx - 1, stride_x):
                sx = i * dx_px

                if self.engine.solid[j][i]:
                    col = "#111827"
                else:
                    if self.vis_mode == "velocity":
                        vel = math.hypot(self.engine.u[j][i], self.engine.v[j][i])
                        col = self._get_palette_color(vel / (max_u * 1.3), "velocity")
                    elif self.vis_mode == "vorticity":
                        omega = self.engine.omega[j][i]
                        norm_om = 0.5 + 0.5 * (omega / 18.0)
                        col = self._get_palette_color(norm_om, "vorticity")
                    elif self.vis_mode == "pressure":
                        p_val = self.engine.pressure[j][i]
                        norm_p = 0.5 + 0.5 * (p_val / 20.0)
                        col = self._get_palette_color(norm_p, "pressure")
                    else: # Streamfunction
                        psi_val = self.engine.psi[j][i]
                        norm_psi = psi_val / (self.engine.u_inf * self.engine.ly)
                        col = self._get_palette_color(norm_psi, "streamfunction")

                self.canvas.create_rectangle(sx, sy, sx + w_tile, sy + h_tile,
                                             fill=col, outline="", width=0)

        # 2. Solid Obstacle Polygons / Boundaries
        for j in range(ny):
            sy = ch - (j + 1) * dy_px
            for i in range(nx):
                sx = i * dx_px
                if self.engine.solid[j][i]:
                    self.canvas.create_rectangle(sx, sy, sx + dx_px + 1, sy + dy_px + 1,
                                                 fill="#0B0F19", outline="#00F0FF", width=1)

        # 3. Vector Quiver Arrows
        if self.show_vectors:
            arrow_step_x = 4
            arrow_step_y = 3
            for j in range(1, ny - 1, arrow_step_y):
                sy = ch - j * dy_px
                for i in range(1, nx - 1, arrow_step_x):
                    if not self.engine.solid[j][i]:
                        sx = i * dx_px
                        u_v = self.engine.u[j][i]
                        v_v = self.engine.v[j][i]
                        mag = math.hypot(u_v, v_v)
                        if mag > 0.05:
                            scale_arrow = 10.0 / max_u
                            ex = sx + u_v * scale_arrow
                            ey = sy - v_v * scale_arrow
                            self.canvas.create_line(sx, sy, ex, ey, fill="#F1F5F9",
                                                    arrow=tk.LAST, arrowshape=(4, 5, 2))

        # 4. Smoke Tracer Streakline Particles
        if self.show_smoke:
            for p in self.engine.particles:
                px = (p.x / self.engine.lx) * cw
                py = ch - (p.y / self.engine.ly) * ch
                rad = 2.0
                alpha_age = 1.0 - (p.age / p.max_age)
                smoke_col = "#00F0FF" if alpha_age > 0.5 else "#94A3B8"
                self.canvas.create_oval(px - rad, py - rad, px + rad, py + rad,
                                        fill=smoke_col, outline="")

        # 5. Virtual Pitot Probe Crosshair
        if self.probe_pos:
            px = (self.probe_pos[0] / self.engine.lx) * cw
            py = ch - (self.probe_pos[1] / self.engine.ly) * ch

            self.canvas.create_line(px - 10, py, px + 10, py, fill="#F43F5E", width=1.5)
            self.canvas.create_line(px, py - 10, px, py + 10, fill="#F43F5E", width=1.5)
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, outline="#F43F5E", width=1.5)

    def _animate(self):
        if self.is_running:
            # Advance fluid Navier-Stokes step
            self.engine.step(dt=0.016, iterations_poisson=8)

            # Update bottom telemetry labels
            self.lbl_telem_cl.config(text=f"CL: {self.engine.cl:+.2f}")
            self.lbl_telem_cd.config(text=f"CD: {self.engine.cd:.3f}")
            ld = self.engine.cl / max(0.001, self.engine.cd)
            self.lbl_telem_ld.config(text=f"L/D: {ld:.1f}")
            self.lbl_telem_cm.config(text=f"CM: {self.engine.cm:+.3f}")

            # Boundary layer separation / stall detection
            if self.alpha_deg > 14.0 or self.alpha_deg < -14.0:
                self.lbl_telem_stall.config(text="STATUS: BOUNDARY LAYER SEPARATION (STALL)", fg="#F43F5E")
            else:
                self.lbl_telem_stall.config(text="STATUS: LAMINAR ATTACHED BOUNDARY", fg="#10B981")

            self._update_probe_readout()

        self._render_canvas()
        self.root.after(30, self._animate)


def main():
    """Main launch entrypoint."""
    root = tk.Tk()
    app = OptiFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
