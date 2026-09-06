"""
AstroEphemeris: Standalone Desktop Solar System Ephemeris & Orbital Mechanics Studio
Python standard library Tkinter with zero external dependencies.
Features:
- 3D Interactive celestial viewport with 6-DOF orbit/pan/zoom camera.
- Inertial Heliocentric and Synodic Rotating CR3BP viewing modes.
- Real-time 4th-order symplectic Yoshida n-body gravitational integrator.
- Post-Newtonian (1PN) General Relativistic perihelion precession rosette.
- Analytical Lagrangian equilibrium points (L1, L2, L3, L4, L5) and Jacobi Hill curves.
- Full orbital elements telemetry (a, e, i, raan, arg_p, nu, speed, period).
- Spacecraft maneuvering thrusters with prograde, retrograde, and normal impulses.
"""

import math
import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

# Ensure local module import works regardless of invocation directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ephemeris import (
    ASTRONOMICAL_UNIT,
    GRAVITATIONAL_CONSTANT,
    MU_SUN,
    SOLAR_MASS,
    EARTH_MASS,
    MOON_MASS,
    MARS_MASS,
    OrbitalElements,
    CartesianState,
    CelestialBody,
    Spacecraft,
    CR3BPModel,
    SolarSystemEngine,
    cartesian_to_orbital_elements
)
from presets import PRESETS, load_preset, get_preset_list, ScenarioPreset


class AstroEphemerisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AstroEphemeris: 3D Solar System Ephemeris & Orbital Mechanics Studio")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)

        # Style & Color Palette
        self.bg_dark = "#0A0D14"
        self.panel_bg = "#111622"
        self.panel_border = "#1E293B"
        self.accent_cyan = "#00E5FF"
        self.accent_gold = "#FFD700"
        self.text_primary = "#F1F5F9"
        self.text_secondary = "#94A3B8"

        self.root.configure(bg=self.bg_dark)

        # Simulation & Engine State
        self.engine = SolarSystemEngine()
        self.cr3bp_model: Optional[CR3BPModel] = None
        self.cr3bp_state: Optional[List[float]] = None
        self.cr3bp_history: List[Tuple[float, float, float]] = []
        self.initial_jacobi: float = 0.0

        # Animation state
        self.is_running = True
        self.sim_speed_multiplier = 1.0
        self.time_step_base = 86400.0 * 0.5 # 12 hours base

        # 3D Camera State
        self.cam_yaw = 45.0      # degrees
        self.cam_pitch = 30.0    # degrees
        self.cam_distance = 2.2 * ASTRONOMICAL_UNIT # meters
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0

        # Mouse drag state
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_button = 0

        # Display Toggles
        self.show_trails = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_orbits = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_lagrange = tk.BooleanVar(value=True)
        self.show_hill_curves = tk.BooleanVar(value=True)

        # Selected Body for Telemetry
        self.selected_target = tk.StringVar(value="Earth")

        # Current Preset
        self.current_preset_key = "inner_solar_system"
        self.view_mode = "inertial" # 'inertial' or 'cr3bp'

        # Background Starfield Coordinates
        self.stars = self._generate_starfield(120)

        # Build GUI Layout
        self._create_ui()

        # Load initial scenario
        self.load_preset_into_engine(self.current_preset_key)

        # Start animation loop
        self.root.after(30, self._animation_tick)

    def _generate_starfield(self, count: int) -> List[Tuple[float, float, float, str]]:
        import random
        random.seed(42)
        stars = []
        colors = ["#FFFFFF", "#D0E0FF", "#FFE8D0", "#A0C8FF"]
        for _ in range(count):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            size = random.uniform(0.7, 1.8)
            color = random.choice(colors)
            stars.append((x, y, size, color))
        return stars

    def _create_ui(self):
        # Top Master Ribbon
        top_bar = tk.Frame(self.root, bg=self.panel_bg, height=48, bd=0, highlightthickness=1, highlightbackground=self.panel_border)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        lbl_title = tk.Label(
            top_bar, text="ASTROEPHEMERIS 3D",
            font=("Helvetica", 13, "bold"), fg=self.accent_cyan, bg=self.panel_bg
        )
        lbl_title.pack(side=tk.LEFT, padx=(16, 8), pady=8)

        lbl_sub = tk.Label(
            top_bar, text="| N-Body Ephemeris, Relativistic Precession & CR3BP Libration Studio",
            font=("Helvetica", 9), fg=self.text_secondary, bg=self.panel_bg
        )
        lbl_sub.pack(side=tk.LEFT, pady=8)

        # Preset selector in ribbon
        lbl_p = tk.Label(top_bar, text="Scenario:", font=("Helvetica", 9, "bold"), fg=self.text_primary, bg=self.panel_bg)
        lbl_p.pack(side=tk.LEFT, padx=(24, 6))

        self.preset_combo = ttk.Combobox(
            top_bar, values=[PRESETS[k].name for k in get_preset_list()],
            state="readonly", width=34
        )
        self.preset_combo.current(0)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)
        self.preset_combo.pack(side=tk.LEFT, padx=4)

        # View Mode Indicator Badge
        self.lbl_mode_badge = tk.Label(
            top_bar, text="MODE: INERTIAL 3D",
            font=("Courier", 9, "bold"), fg="#10B981", bg="#064E3B", padx=8, pady=2
        )
        self.lbl_mode_badge.pack(side=tk.LEFT, padx=16)

        # Simulation Time Readout
        self.lbl_sim_time = tk.Label(
            top_bar, text="Epoch: T+0.00 Days",
            font=("Courier", 10, "bold"), fg=self.accent_gold, bg=self.panel_bg
        )
        self.lbl_sim_time.pack(side=tk.RIGHT, padx=16)

        # Main Workspace: Left Controls, Center Canvas, Right Telemetry
        workspace = tk.Frame(self.root, bg=self.bg_dark)
        workspace.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Control Dock
        left_dock = tk.Frame(workspace, bg=self.panel_bg, width=240, bd=0, highlightthickness=1, highlightbackground=self.panel_border)
        left_dock.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left_dock.pack_propagate(False)

        self._build_controls(left_dock)

        # Right Telemetry Dock
        right_dock = tk.Frame(workspace, bg=self.panel_bg, width=280, bd=0, highlightthickness=1, highlightbackground=self.panel_border)
        right_dock.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right_dock.pack_propagate(False)

        self._build_telemetry(right_dock)

        # Center 3D Orbital Canvas
        canvas_container = tk.Frame(workspace, bg="#05070B", bd=0, highlightthickness=1, highlightbackground=self.panel_border)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="#05070B", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse event bindings for 3D Camera
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag_rotate)
        self.canvas.bind("<ButtonPress-2>", self._on_mouse_down)
        self.canvas.bind("<B2-Motion>", self._on_mouse_drag_pan)
        self.canvas.bind("<ButtonPress-3>", self._on_mouse_down)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag_pan)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel) # macOS / Windows
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.15)) # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.85)) # Linux

    def _build_controls(self, parent: tk.Frame):
        pad_x = 12

        # Section: Simulation Playback
        lbl_sec1 = tk.Label(parent, text="SIMULATION PLAYBACK", font=("Helvetica", 9, "bold"), fg=self.accent_cyan, bg=self.panel_bg)
        lbl_sec1.pack(anchor=tk.W, padx=pad_x, pady=(12, 6))

        btn_row = tk.Frame(parent, bg=self.panel_bg)
        btn_row.pack(fill=tk.X, padx=pad_x, pady=4)

        self.btn_play = tk.Button(
            btn_row, text="PAUSE", font=("Helvetica", 8, "bold"),
            bg="#2563EB", fg="white", width=7, command=self.toggle_play
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, 4))

        btn_step = tk.Button(
            btn_row, text="STEP", font=("Helvetica", 8, "bold"),
            bg="#334155", fg="white", width=6, command=self.step_simulation
        )
        btn_step.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(
            btn_row, text="RESET", font=("Helvetica", 8, "bold"),
            bg="#475569", fg="white", width=6, command=self.reset_scenario
        )
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Simulation Speed Slider
        lbl_spd = tk.Label(parent, text="Time Warp Multiplier:", font=("Helvetica", 8), fg=self.text_secondary, bg=self.panel_bg)
        lbl_spd.pack(anchor=tk.W, padx=pad_x, pady=(8, 2))

        self.scale_speed = tk.Scale(
            parent, from_=0.1, to=20.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg=self.panel_bg, fg=self.text_primary, highlightthickness=0,
            command=self._on_speed_change
        )
        self.scale_speed.set(1.0)
        self.scale_speed.pack(fill=tk.X, padx=pad_x, pady=2)

        # Section: View & Camera Controls
        lbl_sec2 = tk.Label(parent, text="VIEW & CAMERA", font=("Helvetica", 9, "bold"), fg=self.accent_cyan, bg=self.panel_bg)
        lbl_sec2.pack(anchor=tk.W, padx=pad_x, pady=(16, 6))

        btn_top_view = tk.Button(
            parent, text="Top-Down View (Ecliptic)", font=("Helvetica", 8),
            bg="#1E293B", fg=self.text_primary, command=self._set_top_view
        )
        btn_top_view.pack(fill=tk.X, padx=pad_x, pady=2)

        btn_iso_view = tk.Button(
            parent, text="Isometric 3D View", font=("Helvetica", 8),
            bg="#1E293B", fg=self.text_primary, command=self._set_iso_view
        )
        btn_iso_view.pack(fill=tk.X, padx=pad_x, pady=2)

        btn_reset_cam = tk.Button(
            parent, text="Reset Camera Position", font=("Helvetica", 8),
            bg="#1E293B", fg=self.text_primary, command=self._reset_camera
        )
        btn_reset_cam.pack(fill=tk.X, padx=pad_x, pady=2)

        # Section: Visualization Overlays
        lbl_sec3 = tk.Label(parent, text="DISPLAY OVERLAYS", font=("Helvetica", 9, "bold"), fg=self.accent_cyan, bg=self.panel_bg)
        lbl_sec3.pack(anchor=tk.W, padx=pad_x, pady=(16, 6))

        for text, var in [
            ("Orbital Trace Trajectories", self.show_trails),
            ("Kepler Ellipse Outlines", self.show_orbits),
            ("Celestial Body Labels", self.show_labels),
            ("Ecliptic Coordinate Grid", self.show_grid),
            ("Lagrange Points (L1-L5)", self.show_lagrange),
            ("Jacobi Zero-Velocity Curves", self.show_hill_curves),
        ]:
            cb = tk.Checkbutton(
                parent, text=text, variable=var,
                font=("Helvetica", 8), fg=self.text_primary, bg=self.panel_bg,
                selectcolor="#0F172A", activebackground=self.panel_bg, activeforeground=self.accent_cyan
            )
            cb.pack(anchor=tk.W, padx=pad_x, pady=1)

        # Section: Spacecraft Thruster Control Pad
        lbl_sec4 = tk.Label(parent, text="SPACECRAFT THRUSTERS", font=("Helvetica", 9, "bold"), fg=self.accent_cyan, bg=self.panel_bg)
        lbl_sec4.pack(anchor=tk.W, padx=pad_x, pady=(16, 6))

        thrust_grid = tk.Frame(parent, bg=self.panel_bg)
        thrust_grid.pack(fill=tk.X, padx=pad_x, pady=2)

        btn_pro = tk.Button(
            thrust_grid, text="Prograde (+v)", font=("Helvetica", 8, "bold"),
            bg="#059669", fg="white", command=lambda: self._apply_thrust("prograde")
        )
        btn_pro.grid(row=0, column=0, padx=2, pady=2, sticky="ew")

        btn_retro = tk.Button(
            thrust_grid, text="Retrograde (-v)", font=("Helvetica", 8, "bold"),
            bg="#DC2626", fg="white", command=lambda: self._apply_thrust("retrograde")
        )
        btn_retro.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        btn_norm = tk.Button(
            thrust_grid, text="Normal (+z)", font=("Helvetica", 8),
            bg="#475569", fg="white", command=lambda: self._apply_thrust("normal")
        )
        btn_norm.grid(row=1, column=0, padx=2, pady=2, sticky="ew")

        btn_antinorm = tk.Button(
            thrust_grid, text="Anti-Norm (-z)", font=("Helvetica", 8),
            bg="#475569", fg="white", command=lambda: self._apply_thrust("anti_normal")
        )
        btn_antinorm.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        thrust_grid.columnconfigure(0, weight=1)
        thrust_grid.columnconfigure(1, weight=1)

        # Delta-V Magnitude
        lbl_dv = tk.Label(parent, text="Impulse Magnitude (m/s):", font=("Helvetica", 8), fg=self.text_secondary, bg=self.panel_bg)
        lbl_dv.pack(anchor=tk.W, padx=pad_x, pady=(6, 2))

        self.scale_impulse = tk.Scale(
            parent, from_=10.0, to=1000.0, resolution=10.0, orient=tk.HORIZONTAL,
            bg=self.panel_bg, fg=self.text_primary, highlightthickness=0
        )
        self.scale_impulse.set(100.0)
        self.scale_impulse.pack(fill=tk.X, padx=pad_x, pady=2)

    def _build_telemetry(self, parent: tk.Frame):
        pad_x = 12

        lbl_sec = tk.Label(parent, text="ORBITAL TELEMETRY", font=("Helvetica", 9, "bold"), fg=self.accent_cyan, bg=self.panel_bg)
        lbl_sec.pack(anchor=tk.W, padx=pad_x, pady=(12, 6))

        # Target Selector
        lbl_sel = tk.Label(parent, text="Target Body:", font=("Helvetica", 8), fg=self.text_secondary, bg=self.panel_bg)
        lbl_sel.pack(anchor=tk.W, padx=pad_x, pady=(2, 2))

        self.combo_target = ttk.Combobox(
            parent, textvariable=self.selected_target, state="readonly", width=22
        )
        self.combo_target.pack(fill=tk.X, padx=pad_x, pady=(0, 8))

        # Telemetry Display Text Box
        self.txt_telemetry = tk.Text(
            parent, height=26, bg="#080C14", fg="#00FFCC",
            font=("Courier", 8), bd=0, highlightthickness=1, highlightbackground=self.panel_border,
            padx=8, pady=8
        )
        self.txt_telemetry.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=(0, 12))
        self.txt_telemetry.configure(state=tk.DISABLED)

    def _on_speed_change(self, val):
        self.sim_speed_multiplier = float(val)

    def _set_top_view(self):
        self.cam_yaw = 0.0
        self.cam_pitch = 90.0
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0

    def _set_iso_view(self):
        self.cam_yaw = 45.0
        self.cam_pitch = 35.0

    def _reset_camera(self):
        self.cam_yaw = 45.0
        self.cam_pitch = 30.0
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0
        preset = PRESETS.get(self.current_preset_key)
        if preset:
            self.cam_distance = preset.camera_distance

    def _on_mouse_down(self, event):
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_drag_rotate(self, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.cam_yaw = (self.cam_yaw + dx * 0.4) % 360.0
        self.cam_pitch = max(-89.0, min(89.0, self.cam_pitch + dy * 0.4))
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_drag_pan(self, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.cam_pan_x += dx
        self.cam_pan_y += dy
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._zoom(0.85)
        else:
            self._zoom(1.15)

    def _zoom(self, factor: float):
        self.cam_distance *= factor
        if self.view_mode == "inertial":
            self.cam_distance = max(0.1 * ASTRONOMICAL_UNIT, min(50.0 * ASTRONOMICAL_UNIT, self.cam_distance))
        else:
            self.cam_distance = max(0.2, min(20.0, self.cam_distance))

    def _on_preset_change(self, event=None):
        name_to_key = {p.name: p.key for p in PRESETS.values()}
        sel_name = self.preset_combo.get()
        key = name_to_key.get(sel_name, "inner_solar_system")
        self.load_preset_into_engine(key)

    def toggle_play(self):
        self.is_running = not self.is_running
        self.btn_play.configure(
            text="RESUME" if not self.is_running else "PAUSE",
            bg="#059669" if not self.is_running else "#2563EB"
        )

    def step_simulation(self):
        dt = self.time_step_base * self.sim_speed_multiplier
        if self.view_mode == "inertial":
            self.engine.step_yoshida_4th_order(dt)
        else:
            if self.cr3bp_model and self.cr3bp_state:
                # Sub-step RK4
                dt_dim = dt * 0.1
                self.cr3bp_state = list(self.cr3bp_model.step_rk4(tuple(self.cr3bp_state), dt_dim))
                self.cr3bp_history.append((self.cr3bp_state[0], self.cr3bp_state[1], self.cr3bp_state[2]))
                if len(self.cr3bp_history) > 600:
                    self.cr3bp_history.pop(0)
                self.engine.sim_time_sec += dt_dim

    def reset_scenario(self):
        self.load_preset_into_engine(self.current_preset_key)

    def load_preset_into_engine(self, key: str):
        self.current_preset_key = key
        preset = load_preset(key)

        self.view_mode = preset.view_mode
        self.time_step_base = preset.time_step_sec
        self.cam_distance = preset.camera_distance
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0

        if self.view_mode == "inertial":
            self.engine = SolarSystemEngine(
                enable_relativistic=preset.enable_relativistic,
                rel_scale=preset.relativistic_scale
            )
            for body in preset.bodies:
                self.engine.add_body(body)
            if preset.spacecraft:
                self.engine.set_spacecraft(preset.spacecraft)
            self.cr3bp_model = None
            self.cr3bp_state = None
            self.cr3bp_history = []
            self.lbl_mode_badge.configure(text="MODE: INERTIAL 3D", fg="#10B981", bg="#064E3B")
        else:
            # CR3BP Synodic Mode
            self.engine = SolarSystemEngine()
            self.cr3bp_model = CR3BPModel(preset.cr3bp_mu if preset.cr3bp_mu else 0.01215)
            self.cr3bp_state = list(preset.initial_cr3bp_state) if preset.initial_cr3bp_state else [0.83, 0.0, 0.0, 0.0, 0.05, 0.0]
            self.cr3bp_history = [(self.cr3bp_state[0], self.cr3bp_state[1], self.cr3bp_state[2])]
            self.initial_jacobi = self.cr3bp_model.jacobi_constant(*self.cr3bp_state)
            self.lbl_mode_badge.configure(text="MODE: CR3BP SYNODIC", fg="#00E5FF", bg="#0C4A6E")

        # Update target combobox
        if self.view_mode == "inertial":
            targets = list(self.engine.bodies.keys())
            if self.engine.spacecraft:
                targets.append(self.engine.spacecraft.name)
            self.combo_target.configure(values=targets)
            if "Earth" in targets:
                self.selected_target.set("Earth")
            elif targets:
                self.selected_target.set(targets[0])
        else:
            self.combo_target.configure(values=["Spacecraft (S/C)", "Primary (m1)", "Secondary (m2)", "L1", "L2", "L3", "L4", "L5"])
            self.selected_target.set("Spacecraft (S/C)")

    def _apply_thrust(self, direction: str):
        impulse_val = self.scale_impulse.get()
        if self.view_mode == "inertial" and self.engine.spacecraft:
            sc = self.engine.spacecraft
            vx, vy, vz = sc.state.vx, sc.state.vy, sc.state.vz
            v_mag = math.sqrt(vx * vx + vy * vy + vz * vz)
            if v_mag < 1e-6:
                return

            # Unit velocity vector (prograde)
            ux, uy, uz = vx / v_mag, vy / v_mag, vz / v_mag

            if direction == "prograde":
                sc.apply_impulse(ux * impulse_val, uy * impulse_val, uz * impulse_val)
            elif direction == "retrograde":
                sc.apply_impulse(-ux * impulse_val, -uy * impulse_val, -uz * impulse_val)
            elif direction == "normal":
                sc.apply_impulse(0.0, 0.0, impulse_val)
            elif direction == "anti_normal":
                sc.apply_impulse(0.0, 0.0, -impulse_val)

        elif self.view_mode == "cr3bp" and self.cr3bp_state:
            # Dimensionless impulse in rotating frame
            dv = impulse_val * 0.0001
            if direction == "prograde":
                self.cr3bp_state[4] += dv
            elif direction == "retrograde":
                self.cr3bp_state[4] -= dv
            elif direction == "normal":
                self.cr3bp_state[5] += dv
            elif direction == "anti_normal":
                self.cr3bp_state[5] -= dv

    def _project_3d(self, x: float, y: float, z: float, w_canvas: int, h_canvas: int) -> Tuple[float, float, float]:
        """Projects 3D world coordinates into 2D screen coordinates with perspective."""
        yaw_rad = math.radians(self.cam_yaw)
        pitch_rad = math.radians(self.cam_pitch)

        # Yaw rotation around Z axis
        x1 = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
        y1 = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
        z1 = z

        # Pitch rotation around X axis
        x2 = x1
        y2 = y1 * math.cos(pitch_rad) - z1 * math.sin(pitch_rad)
        z2 = y1 * math.sin(pitch_rad) + z1 * math.cos(pitch_rad)

        # Orthographic/isometric depth scaling based on camera distance
        scale = (min(w_canvas, h_canvas) * 0.42) / max(1e-15, self.cam_distance)
        sx = w_canvas * 0.5 + x2 * scale + self.cam_pan_x
        sy = h_canvas * 0.5 - z2 * scale + self.cam_pan_y
        depth = y2

        return (sx, sy, depth)

    def _animation_tick(self):
        if self.is_running:
            self.step_simulation()

        self.render_all()
        self.update_telemetry()
        self.root.after(30, self._animation_tick)

    def render_all(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            return

        # 1. Background Starfield
        for star_x, star_y, star_sz, star_col in self.stars:
            px = w * 0.5 + star_x * w * 0.48
            py = h * 0.5 + star_y * h * 0.48
            self.canvas.create_oval(px, py, px + star_sz, py + star_sz, fill=star_col, outline="")

        if self.view_mode == "inertial":
            self._render_inertial_scene(w, h)
        else:
            self._render_cr3bp_scene(w, h)

        # Draw Coordinate Axes Triad in bottom-left
        self._render_axes_triad(w, h)

    def _render_inertial_scene(self, w: int, h: int):
        # 1. Ecliptic Distance Circles
        if self.show_grid.get():
            for r_au in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                r_m = r_au * ASTRONOMICAL_UNIT
                pts = []
                for step in range(64):
                    theta = (step / 64.0) * 2.0 * math.pi
                    gx = r_m * math.cos(theta)
                    gy = r_m * math.sin(theta)
                    sx, sy, _ = self._project_3d(gx, gy, 0.0, w, h)
                    pts.extend([sx, sy])
                if len(pts) >= 4:
                    self.canvas.create_polygon(pts, fill="", outline="#1E293B", width=1, dash=(2, 4))
                    # Distance label on axis
                    lx, ly, _ = self._project_3d(r_m, 0.0, 0.0, w, h)
                    self.canvas.create_text(lx + 4, ly, text=f"{r_au} AU", fill="#475569", font=("Courier", 7), anchor="w")

        # 2. Keplerian Orbit Outlines
        if self.show_orbits.get():
            for body in self.engine.bodies.values():
                if body.name == "Sun":
                    continue
                a = body.elements.a
                e = body.elements.e
                pts = []
                for step in range(72):
                    nu = (step / 72.0) * 2.0 * math.pi
                    elem = OrbitalElements(a, e, body.elements.i, body.elements.raan, body.elements.arg_p, nu)
                    pos = orbital_elements_to_cartesian(elem, MU_SUN)
                    sx, sy, _ = self._project_3d(pos.x, pos.y, pos.z, w, h)
                    pts.extend([sx, sy])
                if len(pts) >= 4:
                    self.canvas.create_polygon(pts, fill="", outline=body.trail_color, width=1, dash=(3, 3))

        # 3. Trajectory History Trails
        if self.show_trails.get():
            for body in self.engine.bodies.values():
                if len(body.history) > 1:
                    trail_pts = []
                    for hx, hy, hz in body.history:
                        sx, sy, _ = self._project_3d(hx, hy, hz, w, h)
                        trail_pts.extend([sx, sy])
                    if len(trail_pts) >= 4:
                        self.canvas.create_line(trail_pts, fill=body.trail_color, width=1.5)

            if self.engine.spacecraft and len(self.engine.spacecraft.history) > 1:
                sc_pts = []
                for hx, hy, hz in self.engine.spacecraft.history:
                    sx, sy, _ = self._project_3d(hx, hy, hz, w, h)
                    sc_pts.extend([sx, sy])
                if len(sc_pts) >= 4:
                    self.canvas.create_line(sc_pts, fill=self.engine.spacecraft.color, width=2)

        # 4. Render Celestial Bodies
        for body in self.engine.bodies.values():
            sx, sy, _ = self._project_3d(body.state.x, body.state.y, body.state.z, w, h)
            rad = 7 if body.name == "Sun" else (5 if body.name in ["Jupiter", "Saturn"] else 3.5)

            # Glow
            self.canvas.create_oval(sx - rad * 1.6, sy - rad * 1.6, sx + rad * 1.6, sy + rad * 1.6, fill="", outline=body.color, width=1)
            # Solid Body
            self.canvas.create_oval(sx - rad, sy - rad, sx + rad, sy + rad, fill=body.color, outline="#FFFFFF", width=1)

            if self.show_labels.get():
                self.canvas.create_text(sx + rad + 4, sy - 2, text=body.name, fill=self.text_primary, font=("Helvetica", 8, "bold"), anchor="w")

        # 5. Render Spacecraft
        if self.engine.spacecraft:
            sc = self.engine.spacecraft
            sx, sy, _ = self._project_3d(sc.state.x, sc.state.y, sc.state.z, w, h)
            # Diamond marker
            self.canvas.create_polygon(
                sx, sy - 5, sx + 5, sy, sx, sy + 5, sx - 5, sy,
                fill=sc.color, outline="#FFFFFF", width=1
            )
            if self.show_labels.get():
                self.canvas.create_text(sx + 8, sy, text=sc.name, fill=sc.color, font=("Helvetica", 8, "bold"), anchor="w")

    def _render_cr3bp_scene(self, w: int, h: int):
        if not self.cr3bp_model:
            return

        mu = self.cr3bp_model.mu
        # 1. Grid
        if self.show_grid.get():
            for gx_val in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                sx1, sy1, _ = self._project_3d(gx_val, -1.8, 0.0, w, h)
                sx2, sy2, _ = self._project_3d(gx_val, 1.8, 0.0, w, h)
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#1E293B", dash=(2, 4))
            for gy_val in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                sx1, sy1, _ = self._project_3d(-1.8, gy_val, 0.0, w, h)
                sx2, sy2, _ = self._project_3d(1.8, gy_val, 0.0, w, h)
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#1E293B", dash=(2, 4))

        # 2. Zero-Velocity Hill Curves
        if self.show_hill_curves.get() and self.cr3bp_state:
            c_j = self.cr3bp_model.jacobi_constant(*self.cr3bp_state)
            # Sample coarse contour points where 2*Omega(x,y) ~ C_j
            grid_res = 28
            for ix in range(grid_res):
                x_val = -1.6 + (3.2 * ix) / grid_res
                for iy in range(grid_res):
                    y_val = -1.6 + (3.2 * iy) / grid_res
                    om = 2.0 * self.cr3bp_model.effective_potential(x_val, y_val, 0.0)
                    diff = om - c_j
                    if abs(diff) < 0.15:
                        sx, sy, _ = self._project_3d(x_val, y_val, 0.0, w, h)
                        self.canvas.create_oval(sx - 1, sy - 1, sx + 1, sy + 1, fill="#38BDF8", outline="")

        # 3. Lagrangian Libration Points (L1..L5)
        if self.show_lagrange.get():
            for name, (lx, ly, lz) in self.cr3bp_model.lagrange_points.items():
                sx, sy, _ = self._project_3d(lx, ly, lz, w, h)
                col = "#F43F5E" if name in ["L1", "L2", "L3"] else "#10B981"
                # Crosshair marker
                self.canvas.create_line(sx - 4, sy, sx + 4, sy, fill=col, width=1.5)
                self.canvas.create_line(sx, sy - 4, sx, sy + 4, fill=col, width=1.5)
                self.canvas.create_text(sx + 6, sy - 4, text=name, fill=col, font=("Courier", 8, "bold"), anchor="w")

        # 4. Primary Mass (m1) at (-mu, 0, 0)
        sx1, sy1, _ = self._project_3d(-mu, 0.0, 0.0, w, h)
        self.canvas.create_oval(sx1 - 9, sy1 - 9, sx1 + 9, sy1 + 9, fill="#0284C7", outline="#BAE6FD", width=2)
        self.canvas.create_text(sx1, sy1 - 14, text="Primary (m1)", fill="#38BDF8", font=("Helvetica", 8, "bold"))

        # 5. Secondary Mass (m2) at (1 - mu, 0, 0)
        sx2, sy2, _ = self._project_3d(1.0 - mu, 0.0, 0.0, w, h)
        self.canvas.create_oval(sx2 - 4.5, sy2 - 4.5, sx2 + 4.5, sy2 + 4.5, fill="#F59E0B", outline="#FEF3C7", width=1.5)
        self.canvas.create_text(sx2, sy2 - 10, text="Secondary (m2)", fill="#FBBF24", font=("Helvetica", 8, "bold"))

        # 6. Spacecraft Trajectory Trail
        if self.show_trails.get() and len(self.cr3bp_history) > 1:
            pts = []
            for hx, hy, hz in self.cr3bp_history:
                sx, sy, _ = self._project_3d(hx, hy, hz, w, h)
                pts.extend([sx, sy])
            if len(pts) >= 4:
                self.canvas.create_line(pts, fill="#00FFCC", width=1.5)

        # 7. Spacecraft Head
        if self.cr3bp_state:
            sc_x, sc_y, sc_z = self.cr3bp_state[0], self.cr3bp_state[1], self.cr3bp_state[2]
            sc_sx, sc_sy, _ = self._project_3d(sc_x, sc_y, sc_z, w, h)
            self.canvas.create_polygon(
                sc_sx, sc_sy - 5, sc_sx + 5, sc_sy, sc_sx, sc_sy + 5, sc_sx - 5, sc_sy,
                fill="#00FFCC", outline="#FFFFFF", width=1
            )
            self.canvas.create_text(sc_sx + 7, sc_sy, text="S/C", fill="#00FFCC", font=("Helvetica", 8, "bold"), anchor="w")

    def _render_axes_triad(self, w: int, h: int):
        origin_x = 42
        origin_y = h - 42
        axis_len = 26.0

        yaw_rad = math.radians(self.cam_yaw)
        pitch_rad = math.radians(self.cam_pitch)

        axes = [
            (axis_len, 0.0, 0.0, "#EF4444", "X"), # Red
            (0.0, axis_len, 0.0, "#10B981", "Y"), # Green
            (0.0, 0.0, axis_len, "#3B82F6", "Z")  # Blue
        ]

        for ax, ay, az, col, lbl in axes:
            x1 = ax * math.cos(yaw_rad) - ay * math.sin(yaw_rad)
            y1 = ax * math.sin(yaw_rad) + ay * math.cos(yaw_rad)
            z1 = az
            x2 = x1
            y2 = y1 * math.cos(pitch_rad) - z1 * math.sin(pitch_rad)
            z2 = y1 * math.sin(pitch_rad) + z1 * math.cos(pitch_rad)

            end_x = origin_x + x2
            end_y = origin_y - z2

            self.canvas.create_line(origin_x, origin_y, end_x, end_y, fill=col, width=2)
            self.canvas.create_text(end_x + 4, end_y, text=lbl, fill=col, font=("Courier", 7, "bold"))

    def update_telemetry(self):
        # Update Ribbon Epoch
        days = self.engine.sim_time_sec / 86400.0
        self.lbl_sim_time.configure(text=f"Epoch: T+{days:8.2f} Days")

        self.txt_telemetry.configure(state=tk.NORMAL)
        self.txt_telemetry.delete("1.0", tk.END)

        lines = []
        lines.append("=== ASTRODYNAMICS TELEMETRY ===")
        lines.append(f"View Mode : {self.view_mode.upper()}")
        lines.append(f"Sim Time  : {days:.2f} days")
        lines.append(f"Time Step : {self.time_step_base * self.sim_speed_multiplier / 86400.0:.3f} days/step")
        lines.append("")

        if self.view_mode == "inertial":
            target = self.selected_target.get()
            body = self.engine.bodies.get(target)
            if body and body.name != "Sun":
                elem = cartesian_to_orbital_elements(body.state, MU_SUN)
                r_mag = body.state.position_magnitude
                speed = body.state.speed
                period_days = (2.0 * math.pi * math.sqrt(max(1e-15, elem.a ** 3) / MU_SUN)) / 86400.0

                lines.append(f"Target: {body.name.upper()}")
                lines.append(f"Semi-Major (a): {elem.a / ASTRONOMICAL_UNIT:.4f} AU")
                lines.append(f"              : {elem.a / 1000.0:.1f} km")
                lines.append(f"Eccentricity  : {elem.e:.5f}")
                lines.append(f"Inclination   : {math.degrees(elem.i):.2f} deg")
                lines.append(f"RAAN (Omega)  : {math.degrees(elem.raan):.2f} deg")
                lines.append(f"Arg Peri (w)  : {math.degrees(elem.arg_p):.2f} deg")
                lines.append(f"True Anom (nu): {math.degrees(elem.true_anomaly):.2f} deg")
                lines.append(f"Distance (r)  : {r_mag / ASTRONOMICAL_UNIT:.4f} AU")
                lines.append(f"Speed (v)     : {speed / 1000.0:.2f} km/s")
                lines.append(f"Period (T)    : {period_days:.1f} days")
                lines.append("")
                lines.append(f"Periapsis (rp): {elem.periapsis_radius / ASTRONOMICAL_UNIT:.4f} AU")
                lines.append(f"Apoapsis (ra) : {elem.apoapsis_radius / ASTRONOMICAL_UNIT:.4f} AU")

            elif self.engine.spacecraft and target == self.engine.spacecraft.name:
                sc = self.engine.spacecraft
                elem = cartesian_to_orbital_elements(sc.state, MU_SUN)
                r_mag = sc.state.position_magnitude
                speed = sc.state.speed
                # Specific mechanical energy E = v^2/2 - mu/r
                spec_energy = 0.5 * (speed ** 2) - MU_SUN / r_mag

                lines.append(f"Spacecraft: {sc.name.upper()}")
                lines.append(f"Semi-Major (a): {elem.a / ASTRONOMICAL_UNIT:.4f} AU")
                lines.append(f"Eccentricity  : {elem.e:.5f}")
                lines.append(f"Distance (r)  : {r_mag / ASTRONOMICAL_UNIT:.4f} AU")
                lines.append(f"Speed (v)     : {speed / 1000.0:.2f} km/s")
                lines.append(f"Spec Energy E : {spec_energy * 1e-6:.3f} MJ/kg")
                lines.append(f"Delta-V Spent : {sc.delta_v_spent:.1f} m/s")

            # System Energy
            sys_energy = self.engine.compute_system_energy()
            lines.append("")
            lines.append("=== CONSERVATION TELEMETRY ===")
            lines.append(f"Total Energy  : {sys_energy:.3e} J")
            if self.engine.enable_relativistic:
                lines.append("Relativity    : 1PN ACTIVE")
                lines.append(f"Precess Scale : {self.engine.relativistic_scale:.0f}x")

        else: # CR3BP Telemetry
            if self.cr3bp_model and self.cr3bp_state:
                x, y, z, vx, vy, vz = self.cr3bp_state
                v_synodic = math.sqrt(vx * vx + vy * vy + vz * vz)
                c_j = self.cr3bp_model.jacobi_constant(*self.cr3bp_state)
                delta_cj = abs(c_j - self.initial_jacobi) / max(1e-15, abs(self.initial_jacobi))

                lines.append("=== CR3BP SYNODIC FRAME ===")
                lines.append(f"Mass Ratio mu : {self.cr3bp_model.mu:.6f}")
                lines.append(f"Position X    : {x:+.5f}")
                lines.append(f"Position Y    : {y:+.5f}")
                lines.append(f"Position Z    : {z:+.5f}")
                lines.append(f"Synodic Speed : {v_synodic:.5f}")
                lines.append("")
                lines.append("=== JACOBI INTEGRAL ===")
                lines.append(f"Jacobi C_J    : {c_j:.6f}")
                lines.append(f"Initial C_J0  : {self.initial_jacobi:.6f}")
                lines.append(f"Energy Drift  : {delta_cj * 100.0:.4f} %")
                lines.append("")
                lines.append("=== LIBRATION EQUILIBRIA ===")
                for name, (lx, ly, _) in self.cr3bp_model.lagrange_points.items():
                    dist_to_sc = math.hypot(x - lx, y - ly)
                    lines.append(f"{name}: ({lx:+.3f}, {ly:+.3f}) d={dist_to_sc:.3f}")

        self.txt_telemetry.insert(tk.END, "\n".join(lines))
        self.txt_telemetry.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = AstroEphemerisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
