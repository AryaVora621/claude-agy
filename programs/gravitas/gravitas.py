#!/usr/bin/env python3
"""Gravitas 3D - Standalone Desktop N-Body Orbital Mechanics & Celestial Choreography Studio.

Built with standard library Tkinter with zero external dependencies.
Features real-time 3D perspective projection, 4th-order symplectic Yoshida integrator,
Post-Newtonian relativistic perihelion precession, interactive body sling mode,
and live Hamiltonian conservation telemetry HUD.
"""

from __future__ import annotations
import math
import sys
import time
import random
from typing import List, Tuple, Optional, Dict, Any

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: Tkinter is required to run Gravitas Desktop GUI.")
    sys.exit(1)

from .physics import GravitasPhysics, Body, CollisionEvent
from .presets import PRESETS


class Camera3D:
    """Spherical orbital camera with perspective projection and depth-sorting."""

    def __init__(self, distance: float = 65.0, azimuth: float = 45.0, elevation: float = 30.0):
        self.distance = distance
        self.azimuth = azimuth      # Degrees around vertical Z axis
        self.elevation = elevation  # Degrees above orbital plane
        self.target = [0.0, 0.0, 0.0]
        self.fov = 600.0

    def world_to_camera(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Transform Euclidean world coordinate into camera view space."""
        # Translate relative to camera target
        tx = x - self.target[0]
        ty = y - self.target[1]
        tz = z - self.target[2]

        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)

        # Azimuth rotation around Z
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        rx = cos_az * tx + sin_az * ty
        ry = -sin_az * tx + cos_az * ty
        rz = tz

        # Elevation rotation around X-axis
        cos_el, sin_el = math.cos(el_rad), math.sin(el_rad)
        cam_x = rx
        cam_y = cos_el * ry + sin_el * rz
        cam_z = -sin_el * ry + cos_el * rz + self.distance

        return cam_x, cam_y, cam_z

    def project(self, x: float, y: float, z: float, width: int, height: int) -> Tuple[Optional[float], Optional[float], float, float]:
        """Project 3D point to 2D viewport coordinates with depth factor."""
        cam_x, cam_y, cam_z = self.world_to_camera(x, y, z)
        if cam_z <= 1.0:
            return None, None, cam_z, 0.0

        scale = self.fov / cam_z
        screen_x = width / 2.0 + cam_x * scale
        screen_y = height / 2.0 - cam_y * scale
        return screen_x, screen_y, cam_z, scale

    def screen_to_world_ground(self, sx: float, sy: float, width: int, height: int) -> Tuple[float, float]:
        """Invert 2D screen coordinate to approximate (X, Y) point on the Z=0 ground plane."""
        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        cos_el, sin_el = math.cos(el_rad), math.sin(el_rad)

        norm_x = (sx - width / 2.0) / self.fov
        norm_y = -(sy - height / 2.0) / self.fov

        # Ray direction in camera space
        ray_cx = norm_x
        ray_cy = norm_y
        ray_cz = 1.0

        # Rotate ray from camera space back to world space
        # Inverse elevation
        inv_rx = ray_cx
        inv_ry = cos_el * ray_cy - sin_el * ray_cz
        inv_rz = sin_el * ray_cy + cos_el * ray_cz

        # Inverse azimuth
        dir_x = cos_az * inv_rx - sin_az * inv_ry
        dir_y = sin_az * inv_rx + cos_az * inv_ry
        dir_z = inv_rz

        # Camera eye in world coordinates
        cam_dist_z = self.distance * sin_el
        cam_dist_horiz = self.distance * cos_el
        eye_x = self.target[0] - cam_dist_horiz * sin_az
        eye_y = self.target[1] - cam_dist_horiz * cos_az
        eye_z = self.target[2] + cam_dist_z

        # Intersect with Z=0 plane
        if abs(dir_z) < 1e-6:
            t = 0.0
        else:
            t = -eye_z / dir_z

        gx = eye_x + t * dir_x
        gy = eye_y + t * dir_y
        return gx, gy


class GravitasApp:
    """Gravitas 3D Orbital Mechanics Desktop Application."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("Gravitas 3D - N-Body Orbital Mechanics & Celestial Choreography")
        self.root.geometry("1320x860")
        self.root.minsize(1080, 700)
        self.root.configure(bg="#0B0F19")

        # Physics simulation core
        self.sim = GravitasPhysics(g_constant=1.0, softening=0.08)
        self.camera = Camera3D(distance=75.0, azimuth=45.0, elevation=32.0)

        # Simulation states
        self.running = True
        self.integrator = "Yoshida 4th-Order"  # or "Velocity Verlet"
        self.time_warp = 1.0
        self.dt_base = 0.015
        self.show_grid = True
        self.show_trails = True
        self.show_vectors = False
        self.sling_mode = False
        self.sling_start: Optional[Tuple[float, float]] = None
        self.sling_current: Optional[Tuple[float, float]] = None
        self.selected_body: Optional[Body] = None
        self.track_selected = False

        # Interactive creation defaults
        self.spawn_mass = 1.0
        self.spawn_radius = 0.8
        self.spawn_color = "#06B6D4"

        # Mouse interaction cache
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.panning = False

        # Visual explosion spark particles: List of [x, y, z, vx, vy, vz, color, life]
        self.sparks: List[List[Any]] = []

        # Setup styling and widgets
        self._apply_theme()
        self._build_layout()
        self._bind_events()

        # Load initial preset
        self.load_preset("Figure-8 Choreography")

        # Start animation frame loop
        self.last_frame_time = time.time()
        self.fps = 60.0
        self._loop_running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(16, self._update_loop)

    def _apply_theme(self) -> None:
        """Apply modern dark mode styling to ttk controls."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#111827")
        style.configure("Card.TFrame", background="#161F30", relief="flat")
        style.configure("TLabel", background="#111827", foreground="#F9FAFB", font=("Helvetica", 10))
        style.configure("Header.TLabel", background="#111827", foreground="#06B6D4", font=("Helvetica", 11, "bold"))
        style.configure("Muted.TLabel", background="#111827", foreground="#9CA3AF", font=("Helvetica", 9))
        style.configure("Mono.TLabel", background="#161F30", foreground="#F9FAFB", font=("Menlo", 9))
        style.configure("TButton", background="#1F293D", foreground="#F9FAFB", borderwidth=0, font=("Helvetica", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", "#06B6D4"), ("pressed", "#0891B2")], foreground=[("active", "#0B0F19")])
        style.configure("Accent.TButton", background="#06B6D4", foreground="#0B0F19", font=("Helvetica", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#22D3EE"), ("pressed", "#0891B2")])
        style.configure("TCheckbutton", background="#111827", foreground="#F9FAFB", font=("Helvetica", 9))
        style.map("TCheckbutton", background=[("active", "#111827")])
        style.configure("TCombobox", fieldbackground="#1F293D", background="#1F293D", foreground="#F9FAFB")
        style.configure("Horizontal.TScale", background="#111827", troughcolor="#1F293D")

    def _build_layout(self) -> None:
        """Construct top bar, 3D viewport canvas, and control sidebar."""
        # 1. Top Control Ribbon
        top_bar = tk.Frame(self.root, bg="#111827", height=54, highlightthickness=1, highlightbackground="#1F293D")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # Brand Badge
        brand_frame = tk.Frame(top_bar, bg="#111827")
        brand_frame.pack(side=tk.LEFT, padx=16, pady=8)

        logo_lbl = tk.Label(brand_frame, text="GRAVITAS 3D", bg="#06B6D4", fg="#0B0F19", font=("Helvetica", 11, "bold"), padx=8, pady=2)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        sub_lbl = tk.Label(brand_frame, text="N-Body Orbital Dynamics Studio", bg="#111827", fg="#9CA3AF", font=("Helvetica", 10))
        sub_lbl.pack(side=tk.LEFT)

        # Ribbon Action Controls
        ribbon_actions = tk.Frame(top_bar, bg="#111827")
        ribbon_actions.pack(side=tk.RIGHT, padx=16)

        self.btn_play = tk.Button(ribbon_actions, text="❚❚ Pause", bg="#1F293D", fg="#F9FAFB",
                                  activebackground="#06B6D4", activeforeground="#0B0F19",
                                  font=("Helvetica", 9, "bold"), relief="flat", padx=10, pady=4,
                                  command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        btn_step = tk.Button(ribbon_actions, text="▶| Step", bg="#1F293D", fg="#F9FAFB",
                             activebackground="#06B6D4", activeforeground="#0B0F19",
                             font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                             command=self.single_step)
        btn_step.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(ribbon_actions, text="↺ Reset", bg="#1F293D", fg="#F9FAFB",
                              activebackground="#EF4444", activeforeground="#FFFFFF",
                              font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                              command=self.reset_current_preset)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Time Warp controls
        tk.Label(ribbon_actions, text="Speed:", bg="#111827", fg="#9CA3AF", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(12, 4))
        self.scale_speed = ttk.Scale(ribbon_actions, from_=0.1, to=10.0, value=1.0, orient=tk.HORIZONTAL, length=110, command=self._on_speed_change)
        self.scale_speed.pack(side=tk.LEFT, padx=4)
        self.lbl_speed_val = tk.Label(ribbon_actions, text="1.0x", bg="#111827", fg="#06B6D4", font=("Menlo", 9, "bold"), width=5)
        self.lbl_speed_val.pack(side=tk.LEFT, padx=(2, 10))

        # Main Workspace Container
        main_box = tk.Frame(self.root, bg="#0B0F19")
        main_box.pack(fill=tk.BOTH, expand=True)

        # Left: 3D Viewport Canvas
        self.canvas_frame = tk.Frame(main_box, bg="#080C14", highlightthickness=0)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#080C14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Overlay HUD on Canvas (top-left)
        self.hud_label = tk.Label(self.canvas, text="FPS: 60 • Bodies: 3 • Integrator: Yoshida 4th",
                                  bg="#111827", fg="#06B6D4", font=("Menlo", 9), padx=10, pady=4,
                                  highlightthickness=1, highlightbackground="#1F293D")
        self.hud_label.place(x=16, y=16)

        # Right: Multi-deck Sidebar
        sidebar = tk.Frame(main_box, bg="#111827", width=340, highlightthickness=1, highlightbackground="#1F293D")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Scrollable container inside sidebar
        side_canvas = tk.Canvas(sidebar, bg="#111827", highlightthickness=0)
        side_scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=side_canvas.yview)
        self.side_content = tk.Frame(side_canvas, bg="#111827")

        self.side_content.bind(
            "<Configure>",
            lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all"))
        )
        side_canvas.create_window((0, 0), window=self.side_content, anchor="nw", width=320)
        side_canvas.configure(yscrollcommand=side_scroll.set)

        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_sidebar_sections()

    def _build_sidebar_sections(self) -> None:
        """Populate sidebar with presets, physics parameters, spawner, and telemetry."""
        p = self.side_content

        # --- SECTION 1: Presets & Scenario Deck ---
        sec1 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=12)
        sec1.pack(fill=tk.X, padx=12, pady=(12, 6))

        tk.Label(sec1, text="ORBITAL PRESETS", bg="#161F30", fg="#06B6D4", font=("Helvetica", 10, "bold")).pack(anchor="w")

        preset_names = list(PRESETS.keys())
        self.preset_var = tk.StringVar(value="Figure-8 Choreography")
        preset_cb = ttk.Combobox(sec1, textvariable=self.preset_var, values=preset_names, state="readonly")
        preset_cb.pack(fill=tk.X, pady=(6, 8))
        preset_cb.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset_var.get()))

        # Integrator Selection
        tk.Label(sec1, text="Symplectic Integrator:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 9)).pack(anchor="w")
        self.integrator_var = tk.StringVar(value="Yoshida 4th-Order")
        int_cb = ttk.Combobox(sec1, textvariable=self.integrator_var, values=["Yoshida 4th-Order", "Velocity Verlet"], state="readonly")
        int_cb.pack(fill=tk.X, pady=(2, 6))
        int_cb.bind("<<ComboboxSelected>>", lambda e: setattr(self, "integrator", self.integrator_var.get()))

        # Toggles
        self.var_relativity = tk.BooleanVar(value=False)
        chk_rel = tk.Checkbutton(sec1, text="Relativistic Perihelion Precession (1PN)", variable=self.var_relativity,
                                 bg="#161F30", fg="#F9FAFB", selectcolor="#0B0F19", activebackground="#161F30",
                                 font=("Helvetica", 8), command=self._on_toggle_relativity)
        chk_rel.pack(anchor="w", pady=2)

        self.var_collisions = tk.BooleanVar(value=True)
        chk_col = tk.Checkbutton(sec1, text="Inelastic Collisions & Merging", variable=self.var_collisions,
                                 bg="#161F30", fg="#F9FAFB", selectcolor="#0B0F19", activebackground="#161F30",
                                 font=("Helvetica", 8), command=self._on_toggle_collisions)
        chk_col.pack(anchor="w", pady=2)

        # --- SECTION 2: Physics Parameters Sliders ---
        sec2 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=12)
        sec2.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(sec2, text="PHYSICS PARAMETERS", bg="#161F30", fg="#8B5CF6", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # Gravity G
        tk.Label(sec2, text="Gravitational Constant (G):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(6, 0))
        self.scale_g = ttk.Scale(sec2, from_=0.1, to=3.0, value=1.0, orient=tk.HORIZONTAL, command=lambda v: setattr(self.sim, "G", float(v)))
        self.scale_g.pack(fill=tk.X)

        # Softening Epsilon
        tk.Label(sec2, text="Plummer Softening (ε):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        self.scale_soft = ttk.Scale(sec2, from_=0.001, to=0.5, value=0.08, orient=tk.HORIZONTAL, command=lambda v: setattr(self.sim, "softening", float(v)))
        self.scale_soft.pack(fill=tk.X)

        # Trail Length
        tk.Label(sec2, text="Orbital Trail Length:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        self.scale_trails = ttk.Scale(sec2, from_=20, to=300, value=180, orient=tk.HORIZONTAL, command=self._on_trail_length_change)
        self.scale_trails.pack(fill=tk.X)

        # Viewport toggles
        chk_frame = tk.Frame(sec2, bg="#161F30")
        chk_frame.pack(fill=tk.X, pady=(6, 0))

        self.var_grid = tk.BooleanVar(value=True)
        tk.Checkbutton(chk_frame, text="3D Grid Plane", variable=self.var_grid, bg="#161F30", fg="#F9FAFB",
                       selectcolor="#0B0F19", activebackground="#161F30", font=("Helvetica", 8),
                       command=lambda: setattr(self, "show_grid", self.var_grid.get())).pack(side=tk.LEFT)

        self.var_vecs = tk.BooleanVar(value=False)
        tk.Checkbutton(chk_frame, text="Velocity Vectors", variable=self.var_vecs, bg="#161F30", fg="#F9FAFB",
                       selectcolor="#0B0F19", activebackground="#161F30", font=("Helvetica", 8),
                       command=lambda: setattr(self, "show_vectors", self.var_vecs.get())).pack(side=tk.RIGHT)

        # --- SECTION 3: Celestial Body Spawner (Sling Mode) ---
        sec3 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=12)
        sec3.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(sec3, text="INTERACTIVE BODY SPANNER", bg="#161F30", fg="#10B981", font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.btn_sling = tk.Button(sec3, text="⚡ Enter Sling Mode (Click & Drag)", bg="#1F293D", fg="#10B981",
                                   font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=6,
                                   command=self.toggle_sling_mode)
        self.btn_sling.pack(fill=tk.X, pady=(6, 8))

        tk.Label(sec3, text="Launch Mass:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w")
        self.scale_spawn_m = ttk.Scale(sec3, from_=0.1, to=200.0, value=1.0, orient=tk.HORIZONTAL, command=lambda v: setattr(self, "spawn_mass", float(v)))
        self.scale_spawn_m.pack(fill=tk.X)

        tk.Label(sec3, text="Color Swatch:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        color_bar = tk.Frame(sec3, bg="#161F30")
        color_bar.pack(fill=tk.X, pady=4)
        for col in ["#06B6D4", "#8B5CF6", "#10B981", "#F59E0B", "#F43F5E", "#E5E7EB"]:
            btn = tk.Button(color_bar, bg=col, activebackground=col, width=2, height=1, relief="flat",
                            command=lambda c=col: setattr(self, "spawn_color", c))
            btn.pack(side=tk.LEFT, padx=3)

        # --- SECTION 4: Hamiltonian Conservation Telemetry ---
        sec4 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=12)
        sec4.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(sec4, text="HAMILTONIAN CONSERVATION", bg="#161F30", fg="#F59E0B", font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.lbl_telem_ke = tk.Label(sec4, text="Kinetic Energy (T): 0.00 J", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8), anchor="w")
        self.lbl_telem_ke.pack(fill=tk.X, pady=1)

        self.lbl_telem_pe = tk.Label(sec4, text="Potential Energy (U): 0.00 J", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8), anchor="w")
        self.lbl_telem_pe.pack(fill=tk.X, pady=1)

        self.lbl_telem_tot = tk.Label(sec4, text="Total Energy (E): 0.00 J", bg="#161F30", fg="#F9FAFB", font=("Menlo", 8, "bold"), anchor="w")
        self.lbl_telem_tot.pack(fill=tk.X, pady=1)

        self.lbl_telem_drift = tk.Label(sec4, text="Energy Drift ΔE/E: 0.000000", bg="#161F30", fg="#10B981", font=("Menlo", 8, "bold"), anchor="w")
        self.lbl_telem_drift.pack(fill=tk.X, pady=1)

        self.lbl_telem_mom = tk.Label(sec4, text="|L| Angular Momentum: 0.00", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8), anchor="w")
        self.lbl_telem_mom.pack(fill=tk.X, pady=1)

        # --- SECTION 5: Body Inspector Pane ---
        self.sec5 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=12)
        self.sec5.pack(fill=tk.X, padx=12, pady=(6, 12))

        tk.Label(self.sec5, text="CELESTIAL INSPECTOR", bg="#161F30", fg="#EC4899", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.lbl_inspect_name = tk.Label(self.sec5, text="Select any body to inspect", bg="#161F30", fg="#F9FAFB", font=("Helvetica", 9, "bold"), anchor="w")
        self.lbl_inspect_name.pack(fill=tk.X, pady=(4, 2))

        self.lbl_inspect_stats = tk.Label(self.sec5, text="Mass: --\nSpeed: --\nDistance: --", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8), justify=tk.LEFT, anchor="w")
        self.lbl_inspect_stats.pack(fill=tk.X, pady=2)

        inspect_btn_bar = tk.Frame(self.sec5, bg="#161F30")
        inspect_btn_bar.pack(fill=tk.X, pady=(4, 0))

        self.var_track = tk.BooleanVar(value=False)
        self.chk_track = tk.Checkbutton(inspect_btn_bar, text="Track Camera", variable=self.var_track,
                                        bg="#161F30", fg="#F9FAFB", selectcolor="#0B0F19",
                                        activebackground="#161F30", font=("Helvetica", 8),
                                        command=self._on_toggle_track)
        self.chk_track.pack(side=tk.LEFT)

        btn_eject = tk.Button(inspect_btn_bar, text="Eject", bg="#EF4444", fg="#FFFFFF",
                              font=("Helvetica", 8, "bold"), relief="flat", padx=6, pady=2,
                              command=self._on_eject_selected)
        btn_eject.pack(side=tk.RIGHT)

    def _bind_events(self) -> None:
        """Bind mouse drag, wheel zoom, and click selection to 3D canvas."""
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.canvas.bind("<ButtonPress-2>", self._on_right_down)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)

        # Mouse wheel zoom across platforms
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(0.9))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.1))

        # Window resize
        self.canvas.bind("<Configure>", lambda e: self.render_frame())

    def toggle_play(self) -> None:
        """Toggle simulation pause / resume."""
        self.running = not self.running
        self.btn_play.configure(text="❚❚ Pause" if self.running else "▶ Play")

    def single_step(self) -> None:
        """Advance simulation by a single discrete time step while paused."""
        self.running = False
        self.btn_play.configure(text="▶ Play")
        dt = self.dt_base * self.time_warp
        if self.integrator == "Yoshida 4th-Order":
            self.sim.step_yoshida(dt)
        else:
            self.sim.step_verlet(dt)
        self.render_frame()

    def reset_current_preset(self) -> None:
        """Reset current preset back to initial conditions."""
        self.load_preset(self.preset_var.get())

    def load_preset(self, name: str) -> None:
        """Load and initialize a pre-configured orbital choreography."""
        setup_fn = PRESETS.get(name)
        if setup_fn:
            setup_fn(self.sim)
            self.var_relativity.set(self.sim.enable_relativity)
            self.scale_g.set(self.sim.G)
            self.scale_soft.set(self.sim.softening)
            self.selected_body = None
            self.var_track.set(False)
            self.track_selected = False
            self.camera.target = [0.0, 0.0, 0.0]
            # Set default camera distance appropriate for the preset
            if "Sol" in name:
                self.camera.distance = 95.0
            elif "Galactic" in name:
                self.camera.distance = 110.0
            else:
                self.camera.distance = 65.0
            self.render_frame()

    def toggle_sling_mode(self) -> None:
        """Toggle interactive sling launch mode."""
        self.sling_mode = not self.sling_mode
        if self.sling_mode:
            self.btn_sling.configure(bg="#10B981", fg="#0B0F19", text="✓ Sling Active: Click & Drag on Canvas")
        else:
            self.btn_sling.configure(bg="#1F293D", fg="#10B981", text="⚡ Enter Sling Mode (Click & Drag)")
            self.sling_start = None
            self.sling_current = None
        self.render_frame()

    def _on_speed_change(self, val: str) -> None:
        """Update simulation speed factor."""
        self.time_warp = float(val)
        self.lbl_speed_val.configure(text=f"{self.time_warp:.1f}x")

    def _on_trail_length_change(self, val: str) -> None:
        """Update max deque length for all active body trails."""
        length = int(float(val))
        for b in self.sim.bodies:
            current_trail = list(b.trail)
            b.trail = type(b.trail)(current_trail, maxlen=length)

    def _on_toggle_relativity(self) -> None:
        """Toggle post-Newtonian relativistic precession correction."""
        self.sim.enable_relativity = self.var_relativity.get()

    def _on_toggle_collisions(self) -> None:
        """Toggle inelastic collision resolution."""
        self.sim.enable_collisions = self.var_collisions.get()

    def _on_toggle_track(self) -> None:
        """Toggle camera lock on selected body."""
        self.track_selected = self.var_track.get()

    def _on_eject_selected(self) -> None:
        """Remove selected body from simulation."""
        if self.selected_body and self.selected_body.alive:
            self.selected_body.alive = False
            self.selected_body = None
            self.lbl_inspect_name.configure(text="Select any body to inspect")
            self.lbl_inspect_stats.configure(text="Mass: --\nSpeed: --\nDistance: --")
            self.render_frame()

    # --- Mouse & 3D Navigation Handlers ---

    def _on_mouse_down(self, event: tk.Event) -> None:
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        if self.sling_mode:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            gx, gy = self.camera.screen_to_world_ground(event.x, event.y, w, h)
            self.sling_start = (gx, gy)
            self.sling_current = (gx, gy)
            self.render_frame()
            return

        # Check if user clicked on an existing body
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        clicked_body = None
        min_dist = 18.0

        for b in self.sim.bodies:
            if not b.alive:
                continue
            sx, sy, cz, sc = self.camera.project(b.pos[0], b.pos[1], b.pos[2], w, h)
            if sx is not None and sy is not None:
                d = math.hypot(event.x - sx, event.y - sy)
                if d < min_dist:
                    min_dist = d
                    clicked_body = b

        if clicked_body:
            self.selected_body = clicked_body
            self._update_inspector_ui()
            self.render_frame()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.sling_mode and self.sling_start:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            gx, gy = self.camera.screen_to_world_ground(event.x, event.y, w, h)
            self.sling_current = (gx, gy)
            self.render_frame()
            return

        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Orbit camera: horizontal drag rotates azimuth, vertical drag tilts elevation
        self.camera.azimuth = (self.camera.azimuth + dx * 0.5) % 360.0
        self.camera.elevation = max(-85.0, min(85.0, self.camera.elevation - dy * 0.5))
        self.render_frame()

    def _on_mouse_up(self, event: tk.Event) -> None:
        if self.sling_mode and self.sling_start and self.sling_current:
            # Spawn new body with velocity vector proportional to drag
            gx_start, gy_start = self.sling_start
            gx_end, gy_end = self.sling_current

            # Velocity vector = (start - end) * launch_scale
            v_scale = 0.25
            vx = (gx_start - gx_end) * v_scale
            vy = (gy_start - gy_end) * v_scale

            rad = max(0.4, min(2.5, self.spawn_mass ** (1.0 / 3.0) * 0.6))
            new_b = self.sim.add_body(
                name=f"Asteroid-{self.sim.next_body_id}",
                mass=self.spawn_mass,
                radius=rad,
                color=self.spawn_color,
                pos=[gx_start, gy_start, 0.0],
                vel=[vx, vy, 0.0]
            )
            self.selected_body = new_b
            self._update_inspector_ui()

            self.sling_start = None
            self.sling_current = None
            self.render_frame()

    def _on_right_down(self, event: tk.Event) -> None:
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_right_drag(self, event: tk.Event) -> None:
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Pan camera target across screen view plane
        pan_scale = self.camera.distance * 0.0015
        az_rad = math.radians(self.camera.azimuth)
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)

        self.camera.target[0] -= (cos_az * dx - sin_az * dy) * pan_scale
        self.camera.target[1] -= (sin_az * dx + cos_az * dy) * pan_scale
        self.render_frame()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        # macOS vs Windows delta
        factor = 0.9 if (event.delta > 0 or getattr(event, "num", None) == 4) else 1.1
        self._zoom(factor)

    def _zoom(self, factor: float) -> None:
        self.camera.distance = max(10.0, min(300.0, self.camera.distance * factor))
        self.render_frame()

    def _update_inspector_ui(self) -> None:
        """Refresh selected body inspector readout."""
        b = self.selected_body
        if not b or not b.alive:
            self.lbl_inspect_name.configure(text="Select any body to inspect")
            self.lbl_inspect_stats.configure(text="Mass: --\nSpeed: --\nDistance: --")
            return

        self.lbl_inspect_name.configure(text=f"{b.name} (ID #{b.id})")
        sp = b.speed()
        dist = math.sqrt(b.pos[0]**2 + b.pos[1]**2 + b.pos[2]**2)
        ke = b.kinetic_energy()
        self.lbl_inspect_stats.configure(
            text=f"Mass: {b.mass:.2f} M\nRadius: {b.radius:.2f} R\nSpeed: {sp:.3f} AU/s\nDistance: {dist:.2f} AU\nKinetic: {ke:.2f} J"
        )

    # --- Simulation Step and Render Pipeline ---

    def _update_loop(self) -> None:
        """Main game loop executing physics step and graphical refresh."""
        if not self._loop_running:
            return

        now = time.time()
        dt_frame = now - self.last_frame_time
        self.last_frame_time = now
        if dt_frame > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt_frame)

        if self.running:
            # Substep physics for ultra-smooth high-frequency stability
            substeps = 2
            dt_sub = (self.dt_base * self.time_warp) / substeps
            for _ in range(substeps):
                if self.integrator == "Yoshida 4th-Order":
                    self.sim.step_yoshida(dt_sub)
                else:
                    self.sim.step_verlet(dt_sub)

            # Camera tracking
            if self.track_selected and self.selected_body and self.selected_body.alive:
                self.camera.target = list(self.selected_body.pos)

        # Update visual spark particles
        dt_sec = 0.016
        for spk in self.sparks:
            spk[0] += spk[3] * dt_sec
            spk[1] += spk[4] * dt_sec
            spk[2] += spk[5] * dt_sec
            spk[7] -= dt_sec
        self.sparks = [s for s in self.sparks if s[7] > 0.0]

        # Ingest new collision events into spark explosions
        for ev in self.sim.collision_events:
            if ev.age < 0.03:
                rng = random.Random(int(ev.pos[0] * 100))
                for _ in range(24):
                    th = rng.uniform(0, 2 * math.pi)
                    phi = rng.uniform(-math.pi / 2, math.pi / 2)
                    spd = rng.uniform(2.0, 10.0)
                    vx = spd * math.cos(phi) * math.cos(th)
                    vy = spd * math.cos(phi) * math.sin(th)
                    vz = spd * math.sin(phi)
                    self.sparks.append([ev.pos[0], ev.pos[1], ev.pos[2], vx, vy, vz, "#F59E0B", 0.6])

        # Draw frame
        self.render_frame()

        # Update telemetry labels
        if self.running:
            self._update_telemetry()
            if self.selected_body:
                self._update_inspector_ui()

        self.root.after(16, self._update_loop)

    def _update_telemetry(self) -> None:
        """Update top HUD and telemetry indicators."""
        ke = self.sim.kinetic_energy()
        pe = self.sim.potential_energy()
        tot = ke + pe
        drift = self.sim.energy_drift()
        lm = self.sim.angular_momentum()
        l_mag = math.sqrt(lm[0]**2 + lm[1]**2 + lm[2]**2)

        self.lbl_telem_ke.configure(text=f"Kinetic Energy (T): {ke:10.2f} J")
        self.lbl_telem_pe.configure(text=f"Potential Energy (U): {pe:10.2f} J")
        self.lbl_telem_tot.configure(text=f"Total Energy (E): {tot:10.2f} J")

        # Color drift: green if < 0.001, amber if < 0.01, rose otherwise
        drift_col = "#10B981" if drift < 0.001 else ("#F59E0B" if drift < 0.02 else "#F43F5E")
        self.lbl_telem_drift.configure(text=f"Energy Drift ΔE/E: {drift:8.6f}", fg=drift_col)
        self.lbl_telem_mom.configure(text=f"|L| Angular Momentum: {l_mag:8.2f}")

        hud_text = f"FPS: {self.fps:4.1f} • Bodies: {len(self.sim.bodies)} • Time: {self.sim.sim_time:5.1f}s • Drift: {drift:.5f}"
        self.hud_label.configure(text=hud_text)

    def render_frame(self) -> None:
        """Render 3D scene onto Tkinter Canvas with depth-sorting."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        render_queue = []

        # 1. Coordinate Grid Plane (Z = 0)
        if self.show_grid:
            grid_size = 40.0
            step = 10.0
            num_lines = int(grid_size / step)
            grid_col = "#161F30"

            for i in range(-num_lines, num_lines + 1):
                coord = i * step
                # X parallel line
                render_queue.append({
                    "type": "line",
                    "p1": [-grid_size, coord, 0.0],
                    "p2": [grid_size, coord, 0.0],
                    "color": "#1E293B" if coord == 0 else grid_col,
                    "width": 2 if coord == 0 else 1,
                    "depth": self.camera.world_to_camera(0.0, coord, 0.0)[2]
                })
                # Y parallel line
                render_queue.append({
                    "type": "line",
                    "p1": [coord, -grid_size, 0.0],
                    "p2": [coord, grid_size, 0.0],
                    "color": "#1E293B" if coord == 0 else grid_col,
                    "width": 2 if coord == 0 else 1,
                    "depth": self.camera.world_to_camera(coord, 0.0, 0.0)[2]
                })

            # Concentric distance rings
            for r in [10.0, 20.0, 30.0]:
                cam_z = self.camera.world_to_camera(0.0, 0.0, 0.0)[2]
                render_queue.append({
                    "type": "ring",
                    "radius": r,
                    "color": "#192233",
                    "depth": cam_z
                })

        # 2. Celestial Body Trails
        if self.show_trails:
            for b in self.sim.bodies:
                if not b.alive or len(b.trail) < 2:
                    continue
                trail_list = list(b.trail)
                t_len = len(trail_list)
                for k in range(0, t_len - 1, 2):
                    p1 = trail_list[k]
                    p2 = trail_list[min(k + 2, t_len - 1)]
                    mid_z = self.camera.world_to_camera((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5)[2]
                    # Alpha fade simulated by thinner lines towards the tail
                    fraction = k / float(t_len)
                    line_w = 1 if fraction < 0.6 else 2
                    render_queue.append({
                        "type": "trail_segment",
                        "p1": p1,
                        "p2": p2,
                        "color": b.color,
                        "width": line_w,
                        "depth": mid_z
                    })

        # 3. Celestial Bodies & Reticles
        for b in self.sim.bodies:
            if not b.alive:
                continue
            cam_x, cam_y, cam_z = self.camera.world_to_camera(b.pos[0], b.pos[1], b.pos[2])
            render_queue.append({
                "type": "body",
                "body": b,
                "depth": cam_z
            })

            # Velocity vector line
            if self.show_vectors:
                render_queue.append({
                    "type": "vector",
                    "p1": list(b.pos),
                    "p2": [b.pos[0] + b.vel[0] * 1.5, b.pos[1] + b.vel[1] * 1.5, b.pos[2] + b.vel[2] * 1.5],
                    "color": "#10B981",
                    "depth": cam_z - 0.1
                })

        # 4. Explosion Spark Particles
        for spk in self.sparks:
            cam_x, cam_y, cam_z = self.camera.world_to_camera(spk[0], spk[1], spk[2])
            render_queue.append({
                "type": "spark",
                "pos": [spk[0], spk[1], spk[2]],
                "color": spk[6],
                "depth": cam_z
            })

        # Sort all items from back to front (largest depth camera Z first)
        render_queue.sort(key=lambda item: item["depth"], reverse=True)

        # Draw sorted items
        for item in render_queue:
            itype = item["type"]

            if itype == "line" or itype == "trail_segment":
                p1, p2 = item["p1"], item["p2"]
                sx1, sy1, cz1, _ = self.camera.project(p1[0], p1[1], p1[2], w, h)
                sx2, sy2, cz2, _ = self.camera.project(p2[0], p2[1], p2[2], w, h)
                if sx1 is not None and sx2 is not None:
                    self.canvas.create_line(sx1, sy1, sx2, sy2, fill=item["color"], width=item["width"])

            elif itype == "ring":
                r = item["radius"]
                points = []
                steps = 36
                for s in range(steps + 1):
                    theta = (2.0 * math.pi * s) / steps
                    rx = r * math.cos(theta)
                    ry = r * math.sin(theta)
                    sx, sy, cz, _ = self.camera.project(rx, ry, 0.0, w, h)
                    if sx is not None:
                        points.append((sx, sy))
                if len(points) > 2:
                    for s in range(len(points) - 1):
                        self.canvas.create_line(points[s][0], points[s][1], points[s+1][0], points[s+1][1], fill=item["color"], width=1)

            elif itype == "vector":
                p1, p2 = item["p1"], item["p2"]
                sx1, sy1, cz1, _ = self.camera.project(p1[0], p1[1], p1[2], w, h)
                sx2, sy2, cz2, _ = self.camera.project(p2[0], p2[1], p2[2], w, h)
                if sx1 is not None and sx2 is not None:
                    self.canvas.create_line(sx1, sy1, sx2, sy2, fill=item["color"], width=2, arrow=tk.LAST)

            elif itype == "spark":
                spos = item["pos"]
                sx, sy, cz, sc = self.camera.project(spos[0], spos[1], spos[2], w, h)
                if sx is not None and sy is not None:
                    pr = max(1.0, 2.0 * (sc / 10.0))
                    self.canvas.create_oval(sx - pr, sy - pr, sx + pr, sy + pr, fill=item["color"], outline="")

            elif itype == "body":
                b = item["body"]
                sx, sy, cz, sc = self.camera.project(b.pos[0], b.pos[1], b.pos[2], w, h)
                if sx is None or sy is None:
                    continue

                # Projected visual radius
                app_rad = max(2.0, b.radius * (sc / 8.0))

                # Projected ground shadow on Z=0
                if abs(b.pos[2]) > 0.4:
                    gx, gy, gcz, gsc = self.camera.project(b.pos[0], b.pos[1], 0.0, w, h)
                    if gx is not None and gy is not None:
                        g_rad = max(1.0, app_rad * 0.6)
                        self.canvas.create_oval(gx - g_rad, gy - g_rad * 0.5, gx + g_rad, gy + g_rad * 0.5, fill="#0F172A", outline="#1E293B")
                        # Elevation dropline
                        self.canvas.create_line(sx, sy, gx, gy, fill="#334155", dash=(2, 4), width=1)

                # Outer soft radiance glow
                glow_rad = app_rad + 3.0
                self.canvas.create_oval(sx - glow_rad, sy - glow_rad, sx + glow_rad, sy + glow_rad,
                                        fill="", outline=b.color, width=1)

                # Body solid sphere
                self.canvas.create_oval(sx - app_rad, sy - app_rad, sx + app_rad, sy + app_rad,
                                        fill=b.color, outline="#FFFFFF" if b == self.selected_body else "#0B0F19", width=1)

                # Selection reticle
                if b == self.selected_body:
                    r_sz = app_rad + 6.0
                    self.canvas.create_rectangle(sx - r_sz, sy - r_sz, sx + r_sz, sy + r_sz,
                                                 outline="#EC4899", width=1, dash=(3, 3))
                    # Reticle label
                    self.canvas.create_text(sx, sy - r_sz - 8, text=b.name, fill="#EC4899", font=("Helvetica", 8, "bold"))
                elif app_rad > 5.0:
                    self.canvas.create_text(sx, sy + app_rad + 8, text=b.name, fill="#9CA3AF", font=("Helvetica", 7))

        # 5. Draw Interactive Sling Launch Indicator
        if self.sling_mode and self.sling_start and self.sling_current:
            sx1, sy1, _, _ = self.camera.project(self.sling_start[0], self.sling_start[1], 0.0, w, h)
            sx2, sy2, _, _ = self.camera.project(self.sling_current[0], self.sling_current[1], 0.0, w, h)
            if sx1 is not None and sx2 is not None:
                # Drag arrow
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#F59E0B", width=2, dash=(4, 2))
                # Opposite velocity vector (slingshot direction)
                vx_screen = sx1 - (sx2 - sx1)
                vy_screen = sy1 - (sy2 - sy1)
                self.canvas.create_line(sx1, sy1, vx_screen, vy_screen, fill="#10B981", width=3, arrow=tk.LAST)
                # Spawn reticle
                self.canvas.create_oval(sx1 - 6, sy1 - 6, sx1 + 6, sy1 + 6, fill=self.spawn_color, outline="#FFFFFF")

    def on_close(self) -> None:
        """Clean shutdown handler."""
        self._loop_running = False
        self.root.destroy()


def main() -> None:
    """Application entry point."""
    root = tk.Tk()
    app = GravitasApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
