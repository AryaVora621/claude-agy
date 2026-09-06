"""
AeroAcoustics Studio - Standalone Desktop GUI Application
Computational Aeroacoustics, Doppler Wave Propagation, and Supersonic Sonic Boom Simulator.
Zero external dependencies, standard library Tkinter only.
"""

import math
import time
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple

# Ensure local module directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acoustics import (
    AeroAcousticEngine,
    sound_speed_from_temp,
    sound_pressure_level_db,
    mach_angle_rad,
    DEFAULT_SOUND_SPEED
)
from presets import PRESETS, load_preset, get_preset_list


class AeroAcousticsApp:
    """
    Tkinter desktop application for interactive aeroacoustic simulation.
    Features 2D expanding wavefront canvas, Mach cone envelopes,
    virtual microphone oscilloscopes, real-time FFT power spectra,
    and polar directivity diagrams.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AeroAcoustics Studio - Supersonic Doppler & Sonic Boom Simulator")
        self.root.geometry("1280x840")
        self.root.minsize(1024, 700)
        self.root.configure(bg="#060A12")

        # Core physics simulation engine
        self.engine = AeroAcousticEngine(
            base_mach=1.40,
            source_freq=220.0,
            source_amplitude=70.0,
            temp_celsius=15.0
        )

        # Application execution state
        self.is_running = True
        self.last_update_time = time.time()
        self.selected_mic_index = 0
        self.active_preset_key = "concorde_cruise"

        # Interactive canvas dragging state
        self.dragged_mic_index: Optional[int] = None

        # Display toggle flags
        self.show_mach_cone = tk.BooleanVar(value=True)
        self.show_wavefronts = tk.BooleanVar(value=True)
        self.show_rays = tk.BooleanVar(value=True)
        self.show_ground_trail = tk.BooleanVar(value=True)

        self.setup_styles()
        self.build_ui()
        self.load_preset_into_ui(self.active_preset_key)

        # Kick off continuous physics animation loop
        self.root.after(20, self.simulation_loop)

    def setup_styles(self):
        """Configures dark modern ttk widget styling."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background="#0C1220", foreground="#F0F4FC", font=("Segoe UI", 9))
        style.configure("TLabel", background="#0C1220", foreground="#F0F4FC")
        style.configure("TFrame", background="#0C1220")
        style.configure("TLabelframe", background="#0C1220", foreground="#00F0FF", font=("Segoe UI", 9, "bold"))
        style.configure("TLabelframe.Label", background="#0C1220", foreground="#00F0FF")
        style.configure("TButton", background="#141C2E", foreground="#F0F4FC", font=("Segoe UI", 9, "bold"), borderwidth=1)
        style.map("TButton",
                  background=[("active", "#00F0FF"), ("pressed", "#00C0CC")],
                  foreground=[("active", "#000000"), ("pressed", "#000000")])
        style.configure("TCombobox", fieldbackground="#090E18", background="#141C2E", foreground="#FFFFFF")
        style.configure("TCheckbutton", background="#0C1220", foreground="#F0F4FC")

    def build_ui(self):
        """Constructs full multi-pane desktop interface."""
        # Top Header Bar
        header_frame = tk.Frame(self.root, bg="#0A0E18", height=50, bd=1, relief="solid")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        # Brand title
        title_box = tk.Frame(header_frame, bg="#0A0E18")
        title_box.pack(side=tk.LEFT, padx=14, pady=8)

        logo_lbl = tk.Label(title_box, text="M", bg="#00F0FF", fg="#000000",
                            font=("Segoe UI", 12, "bold"), width=2, height=1)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        title_lbl = tk.Label(title_box, text="AeroAcoustics Studio", bg="#0A0E18", fg="#F0F4FC",
                             font=("Segoe UI", 13, "bold"))
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(title_box, text="Computational Doppler & Sonic Boom Laboratory",
                           bg="#0A0E18", fg="#8292B0", font=("Segoe UI", 9))
        sub_lbl.pack(side=tk.LEFT, padx=(10, 0))

        # Header controls (Play/Pause, Step, Reset)
        btn_box = tk.Frame(header_frame, bg="#0A0E18")
        btn_box.pack(side=tk.RIGHT, padx=14)

        self.btn_play = tk.Button(btn_box, text="Pause", bg="#00F0FF", fg="#000000",
                                  font=("Segoe UI", 9, "bold"), relief="flat", padx=12, pady=4,
                                  command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        btn_step = tk.Button(btn_box, text="Step", bg="#141C2E", fg="#F0F4FC",
                             font=("Segoe UI", 9), relief="flat", padx=10, pady=4,
                             command=self.step_simulation)
        btn_step.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(btn_box, text="Reset", bg="#141C2E", fg="#F0F4FC",
                              font=("Segoe UI", 9), relief="flat", padx=10, pady=4,
                              command=self.reset_simulation)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Main Workspace: Left Controls Sidebar + Center Wave Canvas + Right Scope Dock
        main_workspace = tk.Frame(self.root, bg="#060A12")
        main_workspace.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Controls Sidebar (Width 280)
        self.build_controls_sidebar(main_workspace)

        # Center / Right Workspace Frame
        center_workspace = tk.Frame(main_workspace, bg="#060A12")
        center_workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Central 2D Wave Propagation Canvas Stage
        self.build_wave_stage(center_workspace)

        # Bottom Scope Dock (Oscilloscope, FFT, Directivity)
        self.build_bottom_dock(center_workspace)

        # Bottom Status Telemetry Bar
        self.build_telemetry_bar()

    def build_controls_sidebar(self, parent: tk.Frame):
        """Constructs parameters and configuration sidebar."""
        sidebar = tk.Frame(parent, bg="#0C1220", width=300, bd=1, relief="solid")
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        sidebar.pack_propagate(False)

        canvas = tk.Canvas(sidebar, bg="#0C1220", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas, bg="#0C1220")

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_content, anchor="nw", width=285)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        pad_opts = {"padx": 10, "pady": 6, "fill": tk.X}

        # 1. Presets Selection Box
        box_presets = ttk.LabelFrame(scroll_content, text="Aero Presets")
        box_presets.pack(**pad_opts)

        self.preset_combo = ttk.Combobox(box_presets, values=get_preset_list(), state="readonly")
        self.preset_combo.set("concorde_cruise")
        self.preset_combo.pack(fill=tk.X, padx=8, pady=6)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)

        self.lbl_preset_desc = tk.Label(box_presets, text="", bg="#0C1220", fg="#8292B0",
                                        font=("Segoe UI", 8), wraplength=250, justify=tk.LEFT)
        self.lbl_preset_desc.pack(fill=tk.X, padx=8, pady=(0, 6))

        # 2. Flight Kinematics
        box_kinematics = ttk.LabelFrame(scroll_content, text="Flight Kinematics")
        box_kinematics.pack(**pad_opts)

        lbl_mach_row = tk.Frame(box_kinematics, bg="#0C1220")
        lbl_mach_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(lbl_mach_row, text="Mach Number (M):", bg="#0C1220", fg="#8292B0").pack(side=tk.LEFT)
        self.val_mach_lbl = tk.Label(lbl_mach_row, text="1.40 M", bg="#0C1220", fg="#00F0FF",
                                     font=("Segoe UI", 9, "bold"))
        self.val_mach_lbl.pack(side=tk.RIGHT)

        self.slider_mach = tk.Scale(box_kinematics, from_=0.10, to=4.00, resolution=0.02,
                                    orient=tk.HORIZONTAL, bg="#0C1220", fg="#F0F4FC",
                                    highlightthickness=0, troughcolor="#141C2E",
                                    showvalue=False, command=self.on_mach_slider_change)
        self.slider_mach.set(1.40)
        self.slider_mach.pack(fill=tk.X, padx=8, pady=(0, 4))

        tk.Label(box_kinematics, text="Flight Trajectory:", bg="#0C1220", fg="#8292B0").pack(anchor="w", padx=8, pady=(4, 0))
        self.traj_combo = ttk.Combobox(box_kinematics,
                                       values=["straight", "accelerating", "circle", "slalom"],
                                       state="readonly")
        self.traj_combo.set("straight")
        self.traj_combo.pack(fill=tk.X, padx=8, pady=4)
        self.traj_combo.bind("<<ComboboxSelected>>", self.on_trajectory_selected)

        # 3. Acoustic Source Properties
        box_source = ttk.LabelFrame(scroll_content, text="Acoustic Source Properties")
        box_source.pack(**pad_opts)

        tk.Label(box_source, text="Radiation Type:", bg="#0C1220", fg="#8292B0").pack(anchor="w", padx=8, pady=(4, 0))
        self.source_type_combo = ttk.Combobox(box_source,
                                              values=["monopole", "dipole", "quadrupole", "n_wave"],
                                              state="readonly")
        self.source_type_combo.set("n_wave")
        self.source_type_combo.pack(fill=tk.X, padx=8, pady=4)
        self.source_type_combo.bind("<<ComboboxSelected>>", self.on_source_type_selected)

        # Source Frequency
        freq_row = tk.Frame(box_source, bg="#0C1220")
        freq_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(freq_row, text="Base Frequency:", bg="#0C1220", fg="#8292B0").pack(side=tk.LEFT)
        self.val_freq_lbl = tk.Label(freq_row, text="220 Hz", bg="#0C1220", fg="#00F0FF",
                                     font=("Segoe UI", 9, "bold"))
        self.val_freq_lbl.pack(side=tk.RIGHT)

        self.slider_freq = tk.Scale(box_source, from_=50, to=800, resolution=5,
                                    orient=tk.HORIZONTAL, bg="#0C1220", fg="#F0F4FC",
                                    highlightthickness=0, troughcolor="#141C2E",
                                    showvalue=False, command=self.on_freq_slider_change)
        self.slider_freq.set(220)
        self.slider_freq.pack(fill=tk.X, padx=8, pady=(0, 4))

        # Source Amplitude / Overpressure
        amp_row = tk.Frame(box_source, bg="#0C1220")
        amp_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(amp_row, text="Peak Overpressure:", bg="#0C1220", fg="#8292B0").pack(side=tk.LEFT)
        self.val_amp_lbl = tk.Label(amp_row, text="70 Pa", bg="#0C1220", fg="#00F0FF",
                                    font=("Segoe UI", 9, "bold"))
        self.val_amp_lbl.pack(side=tk.RIGHT)

        self.slider_amp = tk.Scale(box_source, from_=10, to=200, resolution=5,
                                   orient=tk.HORIZONTAL, bg="#0C1220", fg="#F0F4FC",
                                   highlightthickness=0, troughcolor="#141C2E",
                                   showvalue=False, command=self.on_amp_slider_change)
        self.slider_amp.set(70)
        self.slider_amp.pack(fill=tk.X, padx=8, pady=(0, 4))

        # 4. Atmospheric Environment
        box_env = ttk.LabelFrame(scroll_content, text="Atmospheric Conditions")
        box_env.pack(**pad_opts)

        temp_row = tk.Frame(box_env, bg="#0C1220")
        temp_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(temp_row, text="Temperature:", bg="#0C1220", fg="#8292B0").pack(side=tk.LEFT)
        self.val_temp_lbl = tk.Label(temp_row, text="15.0 deg C", bg="#0C1220", fg="#00F0FF",
                                     font=("Segoe UI", 9, "bold"))
        self.val_temp_lbl.pack(side=tk.RIGHT)

        self.slider_temp = tk.Scale(box_env, from_=-60, to=45, resolution=1,
                                    orient=tk.HORIZONTAL, bg="#0C1220", fg="#F0F4FC",
                                    highlightthickness=0, troughcolor="#141C2E",
                                    showvalue=False, command=self.on_temp_slider_change)
        self.slider_temp.set(15)
        self.slider_temp.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.lbl_sound_speed = tk.Label(box_env, text="Speed of Sound: 343.0 m/s", bg="#0C1220",
                                        fg="#10B981", font=("Segoe UI", 9, "bold"))
        self.lbl_sound_speed.pack(anchor="w", padx=8, pady=(2, 6))

        # 5. Visual Overlays
        box_overlays = ttk.LabelFrame(scroll_content, text="Visual Overlays")
        box_overlays.pack(**pad_opts)

        ttk.Checkbutton(box_overlays, text="Supersonic Mach Cone Envelope",
                        variable=self.show_mach_cone).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box_overlays, text="Propagating Wavefront Rings",
                        variable=self.show_wavefronts).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box_overlays, text="Acoustic Ray Direction Lines",
                        variable=self.show_rays).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box_overlays, text="Ground Boom Footprint Trail",
                        variable=self.show_ground_trail).pack(anchor="w", padx=8, pady=2)

    def build_wave_stage(self, parent: tk.Frame):
        """Constructs central 2D acoustic wavefront propagation canvas."""
        stage_frame = tk.Frame(parent, bg="#04060A")
        stage_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_wave = tk.Canvas(stage_frame, bg="#04060A", highlightthickness=0)
        self.canvas_wave.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for interactive virtual microphone repositioning
        self.canvas_wave.bind("<Button-1>", self.on_canvas_click)
        self.canvas_wave.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_wave.bind("<ButtonRelease-1>", self.on_canvas_release)

    def build_bottom_dock(self, parent: tk.Frame):
        """Constructs bottom docked multi-channel oscilloscope, FFT, and polar plot."""
        dock_frame = tk.Frame(parent, bg="#080C16", height=230, bd=1, relief="solid")
        dock_frame.pack(side=tk.BOTTOM, fill=tk.X)
        dock_frame.pack_propagate(False)

        # Three side-by-side panes:
        # Pane 1: Pressure Oscilloscope (50% width)
        pane_scope = tk.Frame(dock_frame, bg="#080C16")
        pane_scope.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        scope_header = tk.Frame(pane_scope, bg="#080C16")
        scope_header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(scope_header, text="Microphone Pressure Oscilloscope p(t)", bg="#080C16",
                 fg="#00F0FF", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        # Microphone selector
        self.mic_select_combo = ttk.Combobox(scope_header,
                                             values=["MIC_01_GROUND", "MIC_02_FLYBY", "MIC_03_OVERHEAD"],
                                             state="readonly", width=16)
        self.mic_select_combo.set("MIC_01_GROUND")
        self.mic_select_combo.pack(side=tk.RIGHT)
        self.mic_select_combo.bind("<<ComboboxSelected>>", self.on_mic_select_changed)

        self.canvas_scope = tk.Canvas(pane_scope, bg="#04070E", highlightthickness=0)
        self.canvas_scope.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # Pane 2: FFT Power Spectrum (25% width)
        pane_fft = tk.Frame(dock_frame, bg="#080C16", width=260)
        pane_fft.pack(side=tk.LEFT, fill=tk.BOTH, padx=4, pady=4)
        pane_fft.pack_propagate(False)

        tk.Label(pane_fft, text="Acoustic Spectrum (FFT)", bg="#080C16",
                 fg="#10B981", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self.canvas_fft = tk.Canvas(pane_fft, bg="#04070E", highlightthickness=0)
        self.canvas_fft.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # Pane 3: Polar Directivity Diagram (25% width)
        pane_polar = tk.Frame(dock_frame, bg="#080C16", width=220)
        pane_polar.pack(side=tk.LEFT, fill=tk.BOTH, padx=4, pady=4)
        pane_polar.pack_propagate(False)

        tk.Label(pane_polar, text="Aero Directivity D(theta)", bg="#080C16",
                 fg="#A855F7", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self.canvas_polar = tk.Canvas(pane_polar, bg="#04070E", highlightthickness=0)
        self.canvas_polar.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def build_telemetry_bar(self):
        """Constructs bottom telemetry metric ribbon."""
        telemetry = tk.Frame(self.root, bg="#0A0E18", height=32, bd=1, relief="solid")
        telemetry.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_regime = tk.Label(telemetry, text="REGIME: SUPERSONIC", bg="#0A0E18", fg="#00F0FF",
                                   font=("Segoe UI", 9, "bold"), padx=12)
        self.lbl_regime.pack(side=tk.LEFT)

        self.lbl_cone_angle = tk.Label(telemetry, text="Mach Angle (mu): 45.6 deg", bg="#0A0E18",
                                       fg="#F59E0B", font=("Segoe UI", 9), padx=10)
        self.lbl_cone_angle.pack(side=tk.LEFT)

        self.lbl_spl = tk.Label(telemetry, text="Station SPL: 114.2 dBA", bg="#0A0E18",
                                fg="#10B981", font=("Segoe UI", 9), padx=10)
        self.lbl_spl.pack(side=tk.LEFT)

        self.lbl_peak_p = tk.Label(telemetry, text="Peak Overpressure: 85.4 Pa", bg="#0A0E18",
                                   fg="#F43F5E", font=("Segoe UI", 9), padx=10)
        self.lbl_peak_p.pack(side=tk.LEFT)

        self.lbl_wave_count = tk.Label(telemetry, text="Active Wavefronts: 0", bg="#0A0E18",
                                       fg="#8292B0", font=("Segoe UI", 9), padx=10)
        self.lbl_wave_count.pack(side=tk.RIGHT)

    # UI Event Handlers
    def toggle_play(self):
        self.is_running = not self.is_running
        self.btn_play.configure(
            text="Resume" if not self.is_running else "Pause",
            bg="#141C2E" if not self.is_running else "#00F0FF",
            fg="#F0F4FC" if not self.is_running else "#000000"
        )

    def step_simulation(self):
        self.engine.step(0.02)
        self.render_all()

    def reset_simulation(self):
        self.engine.reset()
        self.render_all()

    def on_preset_selected(self, event=None):
        key = self.preset_combo.get()
        self.active_preset_key = key
        self.load_preset_into_ui(key)

    def load_preset_into_ui(self, key: str):
        preset = load_preset(key)
        self.lbl_preset_desc.configure(text=preset.description)

        self.engine.set_mach(preset.base_mach)
        self.slider_mach.set(preset.base_mach)
        self.val_mach_lbl.configure(text=f"{preset.base_mach:.2f} M")

        self.engine.trajectory = self.engine.trajectory.__class__(preset.trajectory_mode)
        self.traj_combo.set(preset.trajectory_mode)

        self.engine.source_type = preset.source_type
        self.source_type_combo.set(preset.source_type)

        self.engine.source_freq = preset.source_freq
        self.slider_freq.set(preset.source_freq)
        self.val_freq_lbl.configure(text=f"{preset.source_freq:.0f} Hz")

        self.engine.source_amplitude = preset.source_amplitude
        self.slider_amp.set(preset.source_amplitude)
        self.val_amp_lbl.configure(text=f"{preset.source_amplitude:.0f} Pa")

        self.engine.set_temperature(preset.temp_celsius)
        self.slider_temp.set(preset.temp_celsius)
        self.val_temp_lbl.configure(text=f"{preset.temp_celsius:.1f} deg C")
        self.lbl_sound_speed.configure(text=f"Speed of Sound: {self.engine.c_sound:.1f} m/s")

        self.engine.reset()

    def on_mach_slider_change(self, val):
        m = float(val)
        self.engine.set_mach(m)
        self.val_mach_lbl.configure(text=f"{m:.2f} M")

    def on_trajectory_selected(self, event=None):
        mode = self.traj_combo.get()
        self.engine.trajectory = self.engine.trajectory.__class__(mode)
        self.engine.reset()

    def on_source_type_selected(self, event=None):
        st = self.source_type_combo.get()
        self.engine.source_type = st

    def on_freq_slider_change(self, val):
        f = float(val)
        self.engine.source_freq = f
        self.val_freq_lbl.configure(text=f"{f:.0f} Hz")

    def on_amp_slider_change(self, val):
        a = float(val)
        self.engine.source_amplitude = a
        self.val_amp_lbl.configure(text=f"{a:.0f} Pa")

    def on_temp_slider_change(self, val):
        t = float(val)
        self.engine.set_temperature(t)
        self.val_temp_lbl.configure(text=f"{t:.1f} deg C")
        self.lbl_sound_speed.configure(text=f"Speed of Sound: {self.engine.c_sound:.1f} m/s")

    def on_mic_select_changed(self, event=None):
        txt = self.mic_select_combo.get()
        for idx, mic in enumerate(self.engine.microphones):
            if mic.station_id == txt:
                self.selected_mic_index = idx
                break

    # Mouse Canvas Dragging for Virtual Microphones
    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        w = self.canvas_wave.winfo_width()
        h = self.canvas_wave.winfo_height()
        cx = w * 0.5
        cy = h * 0.5
        scale = 0.85
        wx = (sx - cx) / scale
        wy = -(sy - cy) / scale
        return wx, wy

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        w = self.canvas_wave.winfo_width()
        h = self.canvas_wave.winfo_height()
        cx = w * 0.5
        cy = h * 0.5
        scale = 0.85
        sx = cx + wx * scale
        sy = cy - wy * scale
        return sx, sy

    def on_canvas_click(self, event):
        wx, wy = self.screen_to_world(event.x, event.y)
        # Check if clicked near any microphone
        self.dragged_mic_index = None
        for i, mic in enumerate(self.engine.microphones):
            dist = math.hypot(mic.x - wx, mic.y - wy)
            if dist < 30.0:
                self.dragged_mic_index = i
                self.selected_mic_index = i
                self.mic_select_combo.set(mic.station_id)
                break

    def on_canvas_drag(self, event):
        if self.dragged_mic_index is not None:
            wx, wy = self.screen_to_world(event.x, event.y)
            mic = self.engine.microphones[self.dragged_mic_index]
            mic.x = wx
            mic.y = wy

    def on_canvas_release(self, event):
        self.dragged_mic_index = None

    # Main Simulation Loop
    def simulation_loop(self):
        now = time.time()
        dt = min(0.04, now - self.last_update_time)
        self.last_update_time = now

        if self.is_running:
            self.engine.step(dt)

        self.render_all()
        self.update_telemetry()
        self.root.after(25, self.simulation_loop)

    # Rendering Subsystems
    def render_all(self):
        self.render_wave_canvas()
        self.render_oscilloscope()
        self.render_fft_spectrum()
        self.render_polar_directivity()

    def render_wave_canvas(self):
        """Draws 2D wavefront field, aircraft, Mach cone, and microphones."""
        cv = self.canvas_wave
        cv.delete("all")
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        cx = w * 0.5
        cy = h * 0.5

        # 1. Background Cartesian Grid & Horizon Ground Plane
        cv.create_rectangle(0, 0, w, h, fill="#04060A", outline="")

        # Coordinate grid lines every 100 meters
        grid_step = 100.0 * 0.85
        for gx in range(-5, 6):
            sx = cx + gx * grid_step
            cv.create_line(sx, 0, sx, h, fill="#0E1626", width=1)
        for gy in range(-4, 5):
            sy = cy + gy * grid_step
            cv.create_line(0, sy, w, sy, fill="#0E1626", width=1)

        # Ground plane boundary line at y = -220m
        _, g_sy = self.world_to_screen(0, -220.0)
        cv.create_line(0, g_sy, w, g_sy, fill="#1E293B", width=2, dash=(6, 4))
        cv.create_text(60, g_sy - 12, text="GROUND LEVEL (y = -220m)", fill="#475569",
                       font=("Segoe UI", 8, "bold"))

        # 2. Ground Boom Footprint Trail
        if self.show_ground_trail.get():
            for boom in self.engine.ground_boom_events:
                bsx, bsy = self.world_to_screen(boom["x"], -220.0)
                cv.create_line(bsx, bsy - 6, bsx, bsy + 6, fill="#F59E0B", width=2)

        # 3. Expanding Spherical Wavefront Rings
        if self.show_wavefronts.get():
            sim_t = self.engine.sim_time
            c_s = self.engine.c_sound
            for wf in self.engine.wavefronts:
                radius = wf.radius_at(sim_t, c_s)
                if radius <= 1.0:
                    continue

                wsx, wsy = self.world_to_screen(wf.x_emit, wf.y_emit)
                r_screen = radius * 0.85

                # Age-dependent transparency color
                if wf.is_shock:
                    color = "#00F0FF" if wf.source_mach < 2.0 else "#F59E0B"
                else:
                    color = "#38BDF8"

                cv.create_oval(wsx - r_screen, wsy - r_screen, wsx + r_screen, wsy + r_screen,
                               outline=color, width=1)

        # 4. Supersonic Mach Cone Envelope
        if self.show_mach_cone.get():
            cone = self.engine.get_mach_cone_geometry()
            if cone is not None:
                (ax, ay) = cone["apex"]
                (x_up, y_up) = cone["line_upper"][1]
                (x_low, y_low) = cone["line_lower"][1]

                asx, asy = self.world_to_screen(ax, ay)
                up_sx, up_sy = self.world_to_screen(x_up, y_up)
                low_sx, low_sy = self.world_to_screen(x_low, y_low)

                # Mach lines
                cv.create_line(asx, asy, up_sx, up_sy, fill="#F43F5E", width=2)
                cv.create_line(asx, asy, low_sx, low_sy, fill="#F43F5E", width=2)

                # Shock wedge area stipple
                cv.create_polygon(asx, asy, up_sx, up_sy, low_sx, low_sy,
                                  fill="#180C14", outline="")

                # Mach angle text annotation
                mu_deg = cone["mu_deg"]
                cv.create_text(asx - 60, asy, text=f"mu = {mu_deg:.1f} deg", fill="#F43F5E",
                               font=("Segoe UI", 9, "bold"))

        # 5. Acoustic Rays between Aircraft and Microphones
        if self.show_rays.get():
            src_sx, src_sy = self.world_to_screen(self.engine.source_x, self.engine.source_y)
            for mic in self.engine.microphones:
                msx, msy = self.world_to_screen(mic.x, mic.y)
                cv.create_line(src_sx, src_sy, msx, msy, fill="#1E293B", width=1, dash=(3, 3))

        # 6. Virtual Microphone Sensor Nodes
        for i, mic in enumerate(self.engine.microphones):
            msx, msy = self.world_to_screen(mic.x, mic.y)
            is_selected = (i == self.selected_mic_index)

            # Pickup aura
            aura_col = "#10B981" if not is_selected else "#00F0FF"
            cv.create_oval(msx - 14, msy - 14, msx + 14, msy + 14, outline=aura_col, width=1.5)

            # Center sensor diode
            cv.create_oval(msx - 5, msy - 5, msx + 5, msy + 5, fill=aura_col, outline="#FFFFFF")

            # Station label
            label_txt = f"{mic.station_id}\n{mic.spl_dBA:.1f} dBA"
            cv.create_text(msx, msy + 24, text=label_txt, fill="#F0F4FC",
                           font=("Segoe UI", 8), justify=tk.CENTER)

        # 7. Moving Aircraft Silhouette Icon & Velocity Vector
        ac_sx, ac_sy = self.world_to_screen(self.engine.source_x, self.engine.source_y)
        heading = math.atan2(self.engine.source_vy, self.engine.source_vx) if (self.engine.source_vx != 0 or self.engine.source_vy != 0) else 0.0

        # Draw delta-wing aircraft shape rotated to heading
        ac_pts = [
            (22, 0),       # Nose
            (-12, -14),    # Left wingtip
            (-6, 0),       # Trailing edge center
            (-12, 14)      # Right wingtip
        ]

        rot_pts = []
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        for px, py in ac_pts:
            rx = px * cos_h - py * sin_h
            ry = px * sin_h + py * cos_h
            rot_pts.extend([ac_sx + rx, ac_sy - ry])

        cv.create_polygon(rot_pts, fill="#FFFFFF", outline="#00F0FF", width=1.5)

        # Velocity vector forward line
        vel_len = min(60.0, self.engine.current_mach * 25.0)
        cv.create_line(ac_sx, ac_sy, ac_sx + vel_len * cos_h, ac_sy - vel_len * sin_h,
                       fill="#10B981", width=2, arrow=tk.LAST)

    def render_oscilloscope(self):
        """Renders time-domain acoustic pressure waveform p(t)."""
        cv = self.canvas_scope
        cv.delete("all")
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10 or self.selected_mic_index >= len(self.engine.microphones):
            return

        mic = self.engine.microphones[self.selected_mic_index]
        history = mic.pressure_history
        N = len(history)

        # Background grid
        cv.create_rectangle(0, 0, w, h, fill="#04070E", outline="")
        mid_y = h * 0.5
        cv.create_line(0, mid_y, w, mid_y, fill="#141C2E", width=1)

        if N < 2:
            cv.create_text(w * 0.5, mid_y, text="Awaiting Acoustic Wave Arrival...",
                           fill="#475569", font=("Segoe UI", 9))
            return

        # Map pressure: scale max 150 Pa to canvas height
        max_p = max(50.0, mic.peak_overpressure_pa * 1.1)
        scale_y = (h * 0.45) / max_p

        pts = []
        for i in range(N):
            x = (i / (N - 1)) * w
            y = mid_y - history[i] * scale_y
            pts.extend([x, y])

        cv.create_line(pts, fill="#00F0FF", width=1.5)

        # Overpressure readout
        latest_p = history[-1] if history else 0.0
        cv.create_text(w - 10, 14, text=f"p(t) = {latest_p:+.1f} Pa", fill="#00F0FF",
                       font=("Consolas", 9, "bold"), anchor="e")

    def render_fft_spectrum(self):
        """Renders frequency power spectrum bars."""
        cv = self.canvas_fft
        cv.delete("all")
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        cv.create_rectangle(0, 0, w, h, fill="#04070E", outline="")

        freqs, powers = self.engine.compute_fft_spectrum(self.selected_mic_index)
        if not powers:
            cv.create_text(w * 0.5, h * 0.5, text="No Spectral Signal", fill="#475569",
                           font=("Segoe UI", 8))
            return

        # Draw vertical spectrum bars
        num_bars = min(40, len(powers))
        bar_w = w / max(1, num_bars)
        min_db = 20.0
        max_db = 140.0

        for i in range(num_bars):
            val = max(min_db, min(max_db, powers[i]))
            bar_h = ((val - min_db) / (max_db - min_db)) * h
            bx = i * bar_w
            by = h - bar_h

            cv.create_rectangle(bx, by, bx + bar_w - 1, h, fill="#10B981", outline="")

    def render_polar_directivity(self):
        """Renders 360-degree polar sound radiation pattern."""
        cv = self.canvas_polar
        cv.delete("all")
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        cv.create_rectangle(0, 0, w, h, fill="#04070E", outline="")
        cx = w * 0.5
        cy = h * 0.5
        max_r = min(w, h) * 0.42

        # Concentric radar distance circles
        for r_frac in [0.33, 0.66, 1.0]:
            r = max_r * r_frac
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#141C2E", width=1)

        # Crosshair axes
        cv.create_line(cx - max_r, cy, cx + max_r, cy, fill="#141C2E", width=1)
        cv.create_line(cx, cy - max_r, cx, cy + max_r, fill="#141C2E", width=1)

        # Polar pattern points
        pattern = self.engine.compute_directivity_polar_pattern(72)
        pts = []
        for angle, intensity in pattern:
            r = intensity * max_r
            px = cx + r * math.cos(angle)
            py = cy - r * math.sin(angle)
            pts.extend([px, py])

        if len(pts) >= 4:
            cv.create_polygon(pts, fill="#23143A", outline="#A855F7", width=1.5)

    def update_telemetry(self):
        """Updates bottom telemetry HUD values."""
        m = self.engine.current_mach
        if m < 0.8:
            regime = "SUBSONIC"
            reg_col = "#38BDF8"
        elif m < 1.2:
            regime = "TRANSONIC"
            reg_col = "#F59E0B"
        elif m < 3.0:
            regime = "SUPERSONIC"
            reg_col = "#F43F5E"
        else:
            regime = "HYPERSONIC"
            reg_col = "#A855F7"

        self.lbl_regime.configure(text=f"REGIME: {regime} (M={m:.2f})", fg=reg_col)

        mu = mach_angle_rad(m)
        if mu is not None:
            self.lbl_cone_angle.configure(text=f"Mach Angle (mu): {math.degrees(mu):.1f} deg")
        else:
            self.lbl_cone_angle.configure(text="Mach Angle (mu): Subsonic (None)")

        if self.selected_mic_index < len(self.engine.microphones):
            mic = self.engine.microphones[self.selected_mic_index]
            self.lbl_spl.configure(text=f"Station SPL: {mic.spl_dBA:.1f} dBA")
            self.lbl_peak_p.configure(text=f"Peak Overpressure: {mic.peak_overpressure_pa:.1f} Pa")

        self.lbl_wave_count.configure(text=f"Active Wavefronts: {len(self.engine.wavefronts)}")


def main():
    root = tk.Tk()
    app = AeroAcousticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
