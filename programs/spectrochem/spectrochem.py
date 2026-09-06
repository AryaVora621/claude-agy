"""
SpectroChem 3D: Standalone Desktop Computational Chemistry & Spectroscopy Studio.
Python standard library Tkinter with zero external dependencies.

Features:
- 3D perspective molecular viewport with full orbit, pan, and zoom camera.
- Ball-and-Stick, Space-Filling (vdW), Wireframe, and Electrostatic Potential rendering.
- Real-time Conjugate Gradient energy minimization.
- Real-time Velocity Verlet Molecular Dynamics with Berendsen thermostat.
- Normal Mode vibrational displacement animation along harmonic eigenvectors.
- Synthetic Infrared (IR) and Raman vibrational absorption spectrum with Lorentzian bands.
- Interactive peak selection to trigger 3D vibrational animations.
- Dipole moment vector and VSEPR geometry classification HUD.
"""

import math
import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

# Ensure local module import works regardless of invocation directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chem_engine import Molecule, Atom, Bond, ELEMENTS, jacobi_eigenvalue_solver
from molecules import MOLECULE_PRESETS


class SpectroChemApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SpectroChem 3D: Molecular Mechanics & Vibrational Spectroscopy")
        self.root.geometry("1240x820")
        self.root.minsize(1000, 680)

        # Apply dark theme styling
        self.bg_dark = "#070913"
        self.bg_panel = "#0D1122"
        self.bg_card = "#141B34"
        self.border_color = "#1E274A"
        self.accent_cyan = "#00F0FF"
        self.accent_orange = "#FF7700"
        self.accent_gold = "#FFB703"
        self.accent_magenta = "#FF007F"
        self.text_main = "#F1F5F9"
        self.text_dim = "#94A3B8"
        self.font_mono = ("SF Mono", 9)
        self.font_sans = ("Segoe UI", 10)

        self.root.configure(bg=self.bg_dark)

        # Simulation State
        self.molecule: Molecule = MOLECULE_PRESETS["water"]()
        self.sim_mode = "vibration"  # "inspect", "optimize", "md", "vibration"
        self.render_mode = "ball_stick"  # "ball_stick", "space_filling", "wireframe", "esp"
        self.is_running = True
        self.temperature_target = 300.0  # Kelvin
        self.md_dt = 0.6  # femtoseconds

        # 3D Camera Controls
        self.camera_rot_x = 0.35  # Elevation
        self.camera_rot_y = 0.55  # Azimuth
        self.camera_zoom = 85.0   # Scale factor
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0

        # Mouse tracking
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.selected_atom_idx: Optional[int] = None
        self.dragged_atom_idx: Optional[int] = None

        # Spectroscopy State
        self.spectrum_peaks: List[Dict] = []
        self.hovered_peak_idx: Optional[int] = None

        # Build UI and compute initial spectrum
        self.build_ui()
        self.reload_molecule("water")

        # Start animation loop
        self.animate()

    def build_ui(self):
        """Constructs the complete application layout and docked controls."""
        # Top Header Ribbon
        header = tk.Frame(self.root, bg=self.bg_panel, height=48, bd=0, highlightthickness=1, highlightbackground=self.border_color)
        header.pack(side=tk.TOP, fill=tk.X)

        # Brand
        title_box = tk.Frame(header, bg=self.bg_panel)
        title_box.pack(side=tk.LEFT, padx=14, pady=8)

        lbl_logo = tk.Label(title_box, text="SC", font=("Segoe UI", 11, "bold"), fg="#FFF", bg=self.accent_cyan, width=3)
        lbl_logo.pack(side=tk.LEFT, padx=(0, 8))

        lbl_title = tk.Label(title_box, text="SPECTROCHEM 3D", font=("Segoe UI", 11, "bold"), fg=self.text_main, bg=self.bg_panel)
        lbl_title.pack(side=tk.LEFT)

        lbl_sub = tk.Label(title_box, text="Molecular Mechanics & FTIR Spectroscopy", font=self.font_mono, fg=self.text_dim, bg=self.bg_panel)
        lbl_sub.pack(side=tk.LEFT, padx=(12, 0))

        # Preset Molecule Selector
        preset_box = tk.Frame(header, bg=self.bg_panel)
        preset_box.pack(side=tk.LEFT, padx=20)

        tk.Label(preset_box, text="Compound:", font=self.font_mono, fg=self.text_dim, bg=self.bg_panel).pack(side=tk.LEFT, padx=(0, 6))
        self.combo_preset = ttk.Combobox(preset_box, values=[
            "Water (H2O)", "Carbon Dioxide (CO2)", "Methane (CH4)", "Ammonia (NH3)",
            "Benzene (C6H6)", "Ethanol (C2H5OH)", "Caffeine (C8H10N4O2)",
            "Aspirin (C9H8O4)", "Sulfur Hexafluoride (SF6)"
        ], state="readonly", width=22)
        self.combo_preset.current(0)
        self.combo_preset.bind("<<ComboboxSelected>>", self.on_preset_change)
        self.combo_preset.pack(side=tk.LEFT)

        # Play / Pause & Mode Buttons
        btn_box = tk.Frame(header, bg=self.bg_panel)
        btn_box.pack(side=tk.RIGHT, padx=14)

        self.btn_play = tk.Button(btn_box, text="PAUSE", font=self.font_mono, fg="#FFF", bg="#1E274A",
                                  activebackground=self.accent_cyan, bd=0, padx=12, pady=4, cursor="hand2", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_opt = tk.Button(btn_box, text="OPTIMIZE (CG)", font=self.font_mono, fg=self.accent_cyan, bg="#141B34",
                                 activebackground=self.accent_cyan, bd=0, padx=10, pady=4, cursor="hand2", command=self.run_single_opt)
        self.btn_opt.pack(side=tk.LEFT, padx=4)

        self.btn_reset = tk.Button(btn_box, text="RESET CAM", font=self.font_mono, fg=self.text_dim, bg="#141B34",
                                   bd=0, padx=8, pady=4, cursor="hand2", command=self.reset_camera)
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        # Main Workspace (Split: Left Viewport & Right Controls)
        workspace = tk.Frame(self.root, bg=self.bg_dark)
        workspace.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Column: 3D Viewport (Top) + Infrared Spectrum (Bottom)
        left_frame = tk.Frame(workspace, bg=self.bg_dark)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 3D Viewport Frame
        view_container = tk.Frame(left_frame, bg=self.bg_panel, highlightthickness=1, highlightbackground=self.border_color)
        view_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

        # Viewport Header Toolbar
        view_header = tk.Frame(view_container, bg=self.bg_card, height=32)
        view_header.pack(fill=tk.X)

        tk.Label(view_header, text="3D MOLECULAR VIEWPORT", font=("Segoe UI", 8, "bold"), fg=self.accent_cyan, bg=self.bg_card).pack(side=tk.LEFT, padx=10, pady=6)

        # Render Mode Radio Buttons
        self.render_var = tk.StringVar(value="ball_stick")
        modes = [("Ball & Stick", "ball_stick"), ("Space-Filling", "space_filling"), ("Wireframe", "wireframe"), ("ESP Surface", "esp")]
        for text, val in modes:
            rb = tk.Radiobutton(view_header, text=text, value=val, variable=self.render_var,
                                font=self.font_mono, fg=self.text_dim, selectcolor=self.bg_dark,
                                bg=self.bg_card, activebackground=self.bg_card, command=self.on_render_mode_change)
            rb.pack(side=tk.LEFT, padx=6)

        self.canvas_3d = tk.Canvas(view_container, bg="#050711", bd=0, highlightthickness=0, cursor="crosshair")
        self.canvas_3d.pack(fill=tk.BOTH, expand=True)

        # Bind 3D Canvas Mouse Events
        self.canvas_3d.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas_3d.bind("<B1-Motion>", self.on_canvas_drag_rotate)
        self.canvas_3d.bind("<ButtonPress-2>", self.on_canvas_press_pan)
        self.canvas_3d.bind("<B2-Motion>", self.on_canvas_drag_pan)
        self.canvas_3d.bind("<ButtonPress-3>", self.on_canvas_press_pan)
        self.canvas_3d.bind("<B3-Motion>", self.on_canvas_drag_pan)
        self.canvas_3d.bind("<MouseWheel>", self.on_canvas_zoom)
        self.canvas_3d.bind("<Button-4>", lambda e: self.zoom_by(1.15))
        self.canvas_3d.bind("<Button-5>", lambda e: self.zoom_by(0.85))

        # Bottom Frame: Infrared (FTIR) & Raman Spectroscopy
        spec_container = tk.Frame(left_frame, bg=self.bg_panel, height=220, highlightthickness=1, highlightbackground=self.border_color)
        spec_container.pack(side=tk.BOTTOM, fill=tk.X)
        spec_container.pack_propagate(False)

        spec_header = tk.Frame(spec_container, bg=self.bg_card, height=28)
        spec_header.pack(fill=tk.X)

        tk.Label(spec_header, text="CALCULATED FTIR & RAMAN VIBRATIONAL SPECTRUM", font=("Segoe UI", 8, "bold"), fg=self.accent_gold, bg=self.bg_card).pack(side=tk.LEFT, padx=10, pady=4)
        tk.Label(spec_header, text="(Click any peak to animate its 3D normal mode vibration)", font=self.font_mono, fg=self.text_dim, bg=self.bg_card).pack(side=tk.LEFT, padx=8)

        self.canvas_spec = tk.Canvas(spec_container, bg="#080A16", bd=0, highlightthickness=0, cursor="hand2")
        self.canvas_spec.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas_spec.bind("<Button-1>", self.on_spectrum_click)
        self.canvas_spec.bind("<Motion>", self.on_spectrum_motion)

        # Right Column: Telemetry HUD, Simulation Controls & Inspector
        right_frame = tk.Frame(workspace, bg=self.bg_panel, width=340, highlightthickness=1, highlightbackground=self.border_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Simulation Mode Control Card
        card_sim = tk.LabelFrame(right_frame, text=" DYNAMICS & MODE ", font=("Segoe UI", 8, "bold"), fg=self.accent_cyan, bg=self.bg_panel, bd=1)
        card_sim.pack(fill=tk.X, padx=10, pady=8)

        self.sim_mode_var = tk.StringVar(value="vibration")
        sim_modes = [
            ("Normal Mode Vibration", "vibration"),
            ("Molecular Dynamics (MD)", "md"),
            ("Geometry Optimization", "optimize"),
            ("Static Inspection", "inspect")
        ]
        for text, val in sim_modes:
            rb = tk.Radiobutton(card_sim, text=text, value=val, variable=self.sim_mode_var,
                                font=self.font_mono, fg=self.text_main, selectcolor=self.bg_card,
                                bg=self.bg_panel, activebackground=self.bg_panel, command=self.on_sim_mode_change)
            rb.pack(anchor=tk.W, padx=10, pady=2)

        # Sliders for MD & Vibration
        slider_box = tk.Frame(card_sim, bg=self.bg_panel)
        slider_box.pack(fill=tk.X, padx=10, pady=6)

        # Temperature
        tk.Label(slider_box, text="MD Target Temp (K):", font=self.font_mono, fg=self.text_dim, bg=self.bg_panel).grid(row=0, column=0, sticky="w")
        self.lbl_temp = tk.Label(slider_box, text="300 K", font=self.font_mono, fg=self.accent_orange, bg=self.bg_panel)
        self.lbl_temp.grid(row=0, column=1, sticky="e")
        self.scale_temp = tk.Scale(slider_box, from_=10, to=800, orient=tk.HORIZONTAL, showvalue=False,
                                   bg=self.bg_card, highlightthickness=0, bd=0, command=self.on_temp_slider)
        self.scale_temp.set(300)
        self.scale_temp.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        # Vibration Amplitude
        tk.Label(slider_box, text="Vibration Amplitude:", font=self.font_mono, fg=self.text_dim, bg=self.bg_panel).grid(row=2, column=0, sticky="w")
        self.lbl_amp = tk.Label(slider_box, text="0.45 A", font=self.font_mono, fg=self.accent_cyan, bg=self.bg_panel)
        self.lbl_amp.grid(row=2, column=1, sticky="e")
        self.scale_amp = tk.Scale(slider_box, from_=5, to=100, orient=tk.HORIZONTAL, showvalue=False,
                                  bg=self.bg_card, highlightthickness=0, bd=0, command=self.on_amp_slider)
        self.scale_amp.set(45)
        self.scale_amp.grid(row=3, column=0, columnspan=2, sticky="ew")

        # Telemetry Card
        card_tel = tk.LabelFrame(right_frame, text=" ENERGY & PROPERTIES ", font=("Segoe UI", 8, "bold"), fg=self.accent_cyan, bg=self.bg_panel, bd=1)
        card_tel.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_epot = self.add_tel_row(card_tel, "Potential Energy:", "0.00 kcal/mol", 0)
        self.lbl_ekin = self.add_tel_row(card_tel, "Kinetic Energy:", "0.00 kcal/mol", 1)
        self.lbl_ebond = self.add_tel_row(card_tel, "Bond Stretch:", "0.00 kcal/mol", 2)
        self.lbl_eangle = self.add_tel_row(card_tel, "Angle Bend:", "0.00 kcal/mol", 3)
        self.lbl_evdw = self.add_tel_row(card_tel, "vdW + Electrostatic:", "0.00 kcal/mol", 4)
        self.lbl_dipole = self.add_tel_row(card_tel, "Dipole Moment (|mu|):", "0.00 Debye", 5, val_fg=self.accent_gold)
        self.lbl_temp_now = self.add_tel_row(card_tel, "Instantaneous Temp:", "0.0 K", 6, val_fg=self.accent_orange)

        # VSEPR Geometry Card
        card_vsepr = tk.LabelFrame(right_frame, text=" VSEPR & SYMMETRY ", font=("Segoe UI", 8, "bold"), fg=self.accent_cyan, bg=self.bg_panel, bd=1)
        card_vsepr.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_vsepr_steric = self.add_tel_row(card_vsepr, "Steric Number:", "4", 0)
        self.lbl_vsepr_geom = self.add_tel_row(card_vsepr, "Coordination Geom:", "Tetrahedral", 1, val_fg=self.accent_cyan)
        self.lbl_vsepr_lp = self.add_tel_row(card_vsepr, "Estimated Lone Pairs:", "0", 2)

        # Normal Modes Listbox
        card_modes = tk.LabelFrame(right_frame, text=" VIBRATIONAL NORMAL MODES ", font=("Segoe UI", 8, "bold"), fg=self.accent_cyan, bg=self.bg_panel, bd=1)
        card_modes.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        self.mode_listbox = tk.Listbox(card_modes, bg=self.bg_card, fg=self.text_main, font=self.font_mono,
                                       selectbackground=self.accent_cyan, selectforeground="#000", bd=0, highlightthickness=0)
        self.mode_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.mode_listbox.bind("<<ListboxSelect>>", self.on_mode_listbox_select)

    def add_tel_row(self, parent, label_text, default_val, row_idx, val_fg=None):
        if val_fg is None:
            val_fg = self.text_main
        f = tk.Frame(parent, bg=self.bg_panel)
        f.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(f, text=label_text, font=self.font_mono, fg=self.text_dim, bg=self.bg_panel).pack(side=tk.LEFT)
        val_lbl = tk.Label(f, text=default_val, font=self.font_mono, fg=val_fg, bg=self.bg_panel)
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl

    def reload_molecule(self, preset_key: str):
        """Loads a preset molecular structure and computes normal modes."""
        builder = MOLECULE_PRESETS.get(preset_key, MOLECULE_PRESETS["water"])
        self.molecule = builder()
        self.molecule.minimize_geometry(max_steps=120)
        self.molecule.initialize_velocities(self.temperature_target)

        # Compute vibrational normal modes
        modes = self.molecule.compute_vibrational_spectrum()

        # Filter out 6 rigid translational/rotational zero modes
        # Keep real vibrational frequencies
        self.spectrum_peaks = [m for m in modes if m['frequency_cm'] > 40.0]

        # Populate modes listbox
        self.mode_listbox.delete(0, tk.END)
        for i, m in enumerate(self.spectrum_peaks):
            freq = m['frequency_cm']
            intens = m['intensity_km_mol']
            self.mode_listbox.insert(tk.END, f"#{i+1:02d} | {freq:7.1f} cm-1 | IR: {intens:5.1f}")

        if self.spectrum_peaks:
            self.mode_listbox.select_set(0)
            self.molecule.active_mode_idx = self.spectrum_peaks[0]['mode_index'] - 1

        self.update_vsepr_hud()

    def update_vsepr_hud(self):
        """Updates VSEPR coordination geometry data."""
        vsepr = self.molecule.get_vsepr_classification(0)
        self.lbl_vsepr_steric.config(text=vsepr['steric_number'])
        self.lbl_vsepr_geom.config(text=vsepr['geometry'])
        self.lbl_vsepr_lp.config(text=vsepr['lone_pairs'])

    def on_preset_change(self, event=None):
        val = self.combo_preset.get().lower()
        key = "water"
        if "co2" in val: key = "carbon_dioxide"
        elif "methane" in val: key = "methane"
        elif "ammonia" in val: key = "ammonia"
        elif "benzene" in val: key = "benzene"
        elif "ethanol" in val: key = "ethanol"
        elif "caffeine" in val: key = "caffeine"
        elif "aspirin" in val: key = "aspirin"
        elif "sf6" in val: key = "sf6"
        self.reload_molecule(key)

    def on_render_mode_change(self):
        self.render_mode = self.render_var.get()

    def on_sim_mode_change(self):
        self.sim_mode = self.sim_mode_var.get()

    def toggle_play(self):
        self.is_running = not self.is_running
        self.btn_play.config(text="PAUSE" if self.is_running else "PLAY")

    def run_single_opt(self):
        steps = self.molecule.minimize_geometry(max_steps=50)
        self.update_telemetry_hud()

    def reset_camera(self):
        self.camera_rot_x = 0.35
        self.camera_rot_y = 0.55
        self.camera_zoom = 85.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0

    def on_temp_slider(self, val):
        self.temperature_target = float(val)
        self.lbl_temp.config(text=f"{int(self.temperature_target)} K")

    def on_amp_slider(self, val):
        amp = float(val) / 100.0
        self.lbl_amp.config(text=f"{amp:.2f} A")

    def on_canvas_press(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Check atom pick
        w = self.canvas_3d.winfo_width()
        h = self.canvas_3d.winfo_height()
        cx, cy = w / 2.0 + self.camera_pan_x, h / 2.0 + self.camera_pan_y

        closest_atom = None
        min_d = 20.0
        for idx, atom in enumerate(self.molecule.atoms):
            px, py, _ = self.project_3d(atom.x, atom.y, atom.z, cx, cy)
            dist = math.sqrt((event.x - px)**2 + (event.y - py)**2)
            if dist < min_d:
                min_d = dist
                closest_atom = idx

        self.selected_atom_idx = closest_atom
        if event.state & 0x0001:  # Shift held down
            self.dragged_atom_idx = closest_atom
        else:
            self.dragged_atom_idx = None

    def on_canvas_drag_rotate(self, event):
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y

        if self.dragged_atom_idx is not None and self.dragged_atom_idx < len(self.molecule.atoms):
            # Manually pull atom in 3D to distort geometry
            atom = self.molecule.atoms[self.dragged_atom_idx]
            scale = 0.012
            atom.x += dx * scale
            atom.y += dy * scale
            self.molecule.compute_energy_and_forces()
        else:
            self.camera_rot_y += dx * 0.012
            self.camera_rot_x += dy * 0.012
            # Clamp elevation
            self.camera_rot_x = max(-1.5, min(1.5, self.camera_rot_x))

        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_canvas_press_pan(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_canvas_drag_pan(self, event):
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.camera_pan_x += dx
        self.camera_pan_y += dy
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_canvas_zoom(self, event):
        if event.delta > 0:
            self.zoom_by(1.1)
        else:
            self.zoom_by(0.9)

    def zoom_by(self, factor):
        self.camera_zoom = max(20.0, min(300.0, self.camera_zoom * factor))

    def on_mode_listbox_select(self, event):
        sel = self.mode_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.spectrum_peaks):
                mode = self.spectrum_peaks[idx]
                self.molecule.active_mode_idx = mode['mode_index'] - 1
                self.sim_mode_var.set("vibration")
                self.sim_mode = "vibration"

    def on_spectrum_click(self, event):
        """User clicks on a vibrational absorption peak to trigger 3D normal mode vibration."""
        w = self.canvas_spec.winfo_width()
        h = self.canvas_spec.winfo_height()
        if w < 50 or h < 50: return

        click_wn = 400.0 + (event.x - 50) / (w - 70) * (4000.0 - 400.0)

        # Find closest peak
        closest_idx = None
        min_diff = 120.0  # cm^-1 tolerance
        for i, peak in enumerate(self.spectrum_peaks):
            diff = abs(peak['frequency_cm'] - click_wn)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        if closest_idx is not None:
            self.mode_listbox.selection_clear(0, tk.END)
            self.mode_listbox.selection_set(closest_idx)
            self.mode_listbox.see(closest_idx)
            mode = self.spectrum_peaks[closest_idx]
            self.molecule.active_mode_idx = mode['mode_index'] - 1
            self.sim_mode_var.set("vibration")
            self.sim_mode = "vibration"

    def on_spectrum_motion(self, event):
        w = self.canvas_spec.winfo_width()
        h = self.canvas_spec.winfo_height()
        if w < 50 or h < 50: return
        hover_wn = 400.0 + (event.x - 50) / (w - 70) * (4000.0 - 400.0)

        self.hovered_peak_idx = None
        for i, peak in enumerate(self.spectrum_peaks):
            if abs(peak['frequency_cm'] - hover_wn) < 60.0:
                self.hovered_peak_idx = i
                break

    def project_3d(self, x: float, y: float, z: float, cx: float, cy: float) -> Tuple[float, float, float]:
        """Projects 3D Cartesian coordinates onto 2D screen viewport."""
        # Azimuth rotation around Y axis
        cos_y, sin_y = math.cos(self.camera_rot_y), math.sin(self.camera_rot_y)
        x1 = x * cos_y + z * sin_y
        y1 = y
        z1 = -x * sin_y + z * cos_y

        # Elevation rotation around X axis
        cos_x, sin_x = math.cos(self.camera_rot_x), math.sin(self.camera_rot_x)
        x2 = x1
        y2 = y1 * cos_x - z1 * sin_x
        z2 = y1 * sin_x + z1 * cos_x

        # Orthographic/weak perspective projection
        screen_x = cx + x2 * self.camera_zoom
        screen_y = cy - y2 * self.camera_zoom
        return screen_x, screen_y, z2

    def animate(self):
        """Main real-time update and render loop."""
        if self.is_running:
            if self.sim_mode == "vibration":
                amp = float(self.scale_amp.get()) / 100.0
                self.molecule.update_normal_mode_vibration(amplitude=amp, speed=0.10)
                self.molecule.compute_energy_and_forces()
            elif self.sim_mode == "md":
                self.molecule.md_step(dt=self.md_dt, target_temperature=self.temperature_target)
            elif self.sim_mode == "optimize":
                self.molecule.minimize_geometry(max_steps=2)
            else:
                self.molecule.compute_energy_and_forces()

        self.render_3d_viewport()
        self.render_spectrum_canvas()
        self.update_telemetry_hud()

        # 30 FPS update loop
        self.root.after(33, self.animate)

    def render_3d_viewport(self):
        """Renders atoms, bonds, electrostatic potential clouds, and dipole moments in 3D."""
        self.canvas_3d.delete("all")
        w = self.canvas_3d.winfo_width()
        h = self.canvas_3d.winfo_height()
        if w < 20 or h < 20: return

        cx = w / 2.0 + self.camera_pan_x
        cy = h / 2.0 + self.camera_pan_y

        # Sort draw order back-to-front (Painter's algorithm)
        draw_items = []

        # 1. Project Bonds
        for bond in self.molecule.bonds:
            a1 = self.molecule.atoms[bond.a1]
            a2 = self.molecule.atoms[bond.a2]
            p1 = self.project_3d(a1.x, a1.y, a1.z, cx, cy)
            p2 = self.project_3d(a2.x, a2.y, a2.z, cx, cy)
            avg_z = (p1[2] + p2[2]) * 0.5
            draw_items.append(('bond', avg_z, p1, p2, bond.order))

        # 2. Project Atoms
        for idx, atom in enumerate(self.molecule.atoms):
            p = self.project_3d(atom.x, atom.y, atom.z, cx, cy)
            draw_items.append(('atom', p[2], p, atom, idx))

        # Sort by depth z (ascending, furthest away drawn first)
        draw_items.sort(key=lambda item: item[1])

        # Draw items
        for item in draw_items:
            kind = item[0]
            if kind == 'bond':
                _, _, p1, p2, order = item
                self.draw_bond(p1, p2, order)
            elif kind == 'atom':
                _, _, p, atom, idx = item
                self.draw_atom(p, atom, idx)

        # Draw Dipole Moment 3D Vector
        mux, muy, muz, mu_mag = self.molecule.get_dipole_moment()
        if mu_mag > 0.05:
            # Draw vector starting at molecule center of mass
            tot_m = sum(a.mass for a in self.molecule.atoms)
            com_x = sum(a.mass * a.x for a in self.molecule.atoms) / tot_m
            com_y = sum(a.mass * a.y for a in self.molecule.atoms) / tot_m
            com_z = sum(a.mass * a.z for a in self.molecule.atoms) / tot_m

            p_com = self.project_3d(com_x, com_y, com_z, cx, cy)
            p_tip = self.project_3d(com_x + mux * 0.6, com_y + muy * 0.6, com_z + muz * 0.6, cx, cy)

            self.canvas_3d.create_line(p_com[0], p_com[1], p_tip[0], p_tip[1], fill=self.accent_gold,
                                       width=2.5, arrow=tk.LAST, arrowshape=(10, 12, 5))
            self.canvas_3d.create_text(p_tip[0] + 8, p_tip[1] - 8, text=f"mu: {mu_mag:.2f} D",
                                       fill=self.accent_gold, font=self.font_mono)

    def draw_bond(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], order: float):
        """Draws chemical bond lines with single, double, or triple representation."""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        if self.render_mode == "wireframe":
            self.canvas_3d.create_line(x1, y1, x2, y2, fill="#718096", width=1.5)
            return

        if order >= 1.8:
            # Double bond: parallel offset lines
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-4: return
            nx = -dy / length * 3.5
            ny = dx / length * 3.5
            self.canvas_3d.create_line(x1 + nx, y1 + ny, x2 + nx, y2 + ny, fill="#A0AEC0", width=2.5)
            self.canvas_3d.create_line(x1 - nx, y1 - ny, x2 - nx, y2 - ny, fill="#A0AEC0", width=2.5)
        elif order >= 1.4:
            # Aromatic bond: solid + dashed
            self.canvas_3d.create_line(x1, y1, x2, y2, fill="#A0AEC0", width=3)
        else:
            # Single bond
            self.canvas_3d.create_line(x1, y1, x2, y2, fill="#718096", width=3.5)

    def draw_atom(self, p: Tuple[float, float, float], atom: Atom, idx: int):
        """Draws atomic sphere with CPK color, radial shading, and labels."""
        x, y = p[0], p[1]

        if self.render_mode == "space_filling":
            r = atom.vdw_radius * (self.camera_zoom * 0.35)
        elif self.render_mode == "wireframe":
            r = 4.0
        else:  # ball_stick or esp
            r = max(8.0, atom.cov_radius * (self.camera_zoom * 0.30))

        # ESP coloring mode: negative (red) to positive (blue)
        color = atom.color
        if self.render_mode == "esp":
            q = atom.charge
            if q < -0.05:
                color = "#FF2244"
            elif q > 0.05:
                color = "#0088FF"
            else:
                color = "#88AA99"

        # Highlight if selected
        outline_color = "#FFFFFF" if idx == self.selected_atom_idx else "#1E274A"
        outline_width = 2.0 if idx == self.selected_atom_idx else 1.0

        # Draw sphere
        self.canvas_3d.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=outline_color, width=outline_width)

        # Draw pseudo-3D light reflection highlight
        if r > 10.0 and self.render_mode != "wireframe":
            hr = r * 0.35
            hx = x - r * 0.3
            hy = y - r * 0.3
            self.canvas_3d.create_oval(hx - hr, hy - hr, hx + hr, hy + hr, fill="#FFFFFF", outline="", stipple="gray50")

        # Element text label
        if self.render_mode != "wireframe" and r > 7.0:
            text_color = "#000000" if atom.element in ['H', 'S'] else "#FFFFFF"
            self.canvas_3d.create_text(x, y, text=atom.element, fill=text_color, font=("Segoe UI", int(max(7, r * 0.65)), "bold"))

    def render_spectrum_canvas(self):
        """Renders calculated FTIR absorption spectrum with Lorentzian bands and interactive markers."""
        self.canvas_spec.delete("all")
        w = self.canvas_spec.winfo_width()
        h = self.canvas_spec.winfo_height()
        if w < 60 or h < 60: return

        # Spectral axis bounds
        wn_min, wn_max = 400.0, 4000.0
        left_pad = 55
        right_pad = 20
        top_pad = 25
        bottom_pad = 30
        plot_w = w - left_pad - right_pad
        plot_h = h - top_pad - bottom_pad

        # Background grid and functional group zones
        # 1. Fingerprint region (400 - 1500 cm-1)
        x_fp_end = left_pad + ((1500.0 - wn_min) / (wn_max - wn_min)) * plot_w
        self.canvas_spec.create_rectangle(left_pad, top_pad, x_fp_end, h - bottom_pad, fill="#0B0F20", outline="")
        self.canvas_spec.create_text((left_pad + x_fp_end) * 0.5, top_pad + 10, text="Fingerprint (400-1500)", fill="#4A5578", font=self.font_mono)

        # 2. Double bond region (1500 - 2000 cm-1)
        x_db_end = left_pad + ((2000.0 - wn_min) / (wn_max - wn_min)) * plot_w
        self.canvas_spec.create_rectangle(x_fp_end, top_pad, x_db_end, h - bottom_pad, fill="#0F1428", outline="")
        self.canvas_spec.create_text((x_fp_end + x_db_end) * 0.5, top_pad + 10, text="C=C, C=O", fill="#4A5578", font=self.font_mono)

        # 3. Triple bond region (2000 - 2500 cm-1)
        x_tb_end = left_pad + ((2500.0 - wn_min) / (wn_max - wn_min)) * plot_w
        self.canvas_spec.create_rectangle(x_db_end, top_pad, x_tb_end, h - bottom_pad, fill="#0B0F20", outline="")

        # 4. X-H stretch region (2500 - 4000 cm-1)
        self.canvas_spec.create_rectangle(x_tb_end, top_pad, left_pad + plot_w, h - bottom_pad, fill="#0F1428", outline="")
        self.canvas_spec.create_text((x_tb_end + left_pad + plot_w) * 0.5, top_pad + 10, text="O-H, N-H, C-H Stretch", fill="#4A5578", font=self.font_mono)

        # Draw Grid Axis Lines
        self.canvas_spec.create_line(left_pad, h - bottom_pad, left_pad + plot_w, h - bottom_pad, fill=self.border_color, width=1)
        self.canvas_spec.create_line(left_pad, top_pad, left_pad, h - bottom_pad, fill=self.border_color, width=1)

        # Wavenumber ticks
        for wn in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
            x = left_pad + ((wn - wn_min) / (wn_max - wn_min)) * plot_w
            self.canvas_spec.create_line(x, h - bottom_pad, x, h - bottom_pad + 5, fill=self.text_dim)
            self.canvas_spec.create_text(x, h - bottom_pad + 14, text=str(wn), fill=self.text_dim, font=self.font_mono)

        self.canvas_spec.create_text(left_pad + plot_w * 0.5, h - 6, text="Wavenumber (cm-1)", fill=self.text_dim, font=self.font_mono)
        self.canvas_spec.create_text(20, top_pad + plot_h * 0.5, text="Abs", fill=self.text_dim, font=self.font_mono)

        # Compute synthetic Lorentzian broadened curve
        num_samples = 300
        gamma = 35.0  # HWHM peak width in cm-1
        max_intensity = max([p['intensity_km_mol'] for p in self.spectrum_peaks], default=10.0)
        max_intensity = max(10.0, max_intensity)

        spectrum_pts = []
        for s in range(num_samples):
            wn = wn_min + (s / (num_samples - 1)) * (wn_max - wn_min)
            # Sum Lorentzians: I * gamma^2 / ((wn - wn0)^2 + gamma^2)
            absorbance = 0.0
            for peak in self.spectrum_peaks:
                p_wn = peak['frequency_cm']
                p_i = peak['intensity_km_mol']
                absorbance += (p_i * gamma**2) / ((wn - p_wn)**2 + gamma**2)

            # Map to screen
            px = left_pad + (s / (num_samples - 1)) * plot_w
            norm_abs = min(1.0, absorbance / (max_intensity * 1.6))
            py = (h - bottom_pad) - norm_abs * (plot_h * 0.85)
            spectrum_pts.append((px, py))

        # Draw spectrum continuous line
        for s in range(len(spectrum_pts) - 1):
            x1, y1 = spectrum_pts[s]
            x2, y2 = spectrum_pts[s + 1]
            self.canvas_spec.create_line(x1, y1, x2, y2, fill=self.accent_cyan, width=1.8)

        # Draw peak stick lines and labels
        active_mode_index = (self.molecule.active_mode_idx + 1) if self.molecule.active_mode_idx is not None else -1
        for i, peak in enumerate(self.spectrum_peaks):
            p_wn = peak['frequency_cm']
            p_i = peak['intensity_km_mol']
            if p_wn < wn_min or p_wn > wn_max: continue

            px = left_pad + ((p_wn - wn_min) / (wn_max - wn_min)) * plot_w
            stick_h = min(plot_h * 0.85, (p_i / (max_intensity * 1.6)) * plot_h * 0.85)
            py = (h - bottom_pad) - stick_h

            is_active = (peak['mode_index'] == active_mode_index)
            is_hovered = (i == self.hovered_peak_idx)

            color = self.accent_magenta if is_active else (self.accent_gold if is_hovered else self.accent_orange)
            width = 2.5 if (is_active or is_hovered) else 1.2

            # Stick line
            self.canvas_spec.create_line(px, h - bottom_pad, px, py, fill=color, width=width)
            # Dot on peak tip
            self.canvas_spec.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, outline="")

            # Peak label
            if is_active or is_hovered or (p_i > max_intensity * 0.3):
                self.canvas_spec.create_text(px, py - 8, text=f"{p_wn:.0f}", fill=color, font=self.font_mono)

    def update_telemetry_hud(self):
        """Updates numerical energy values and physical properties."""
        self.lbl_epot.config(text=f"{self.molecule.e_potential:8.2f} kcal/mol")
        self.lbl_ekin.config(text=f"{self.molecule.e_kinetic:8.2f} kcal/mol")
        self.lbl_ebond.config(text=f"{self.molecule.e_bond:8.2f} kcal/mol")
        self.lbl_eangle.config(text=f"{self.molecule.e_angle:8.2f} kcal/mol")
        self.lbl_evdw.config(text=f"{(self.molecule.e_vdw + self.molecule.e_coulomb):8.2f} kcal/mol")

        _, _, _, mu_mag = self.molecule.get_dipole_moment()
        self.lbl_dipole.config(text=f"{mu_mag:6.2f} Debye")
        self.lbl_temp_now.config(text=f"{self.molecule.temperature:6.1f} K")


def main():
    root = tk.Tk()
    app = SpectroChemApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
