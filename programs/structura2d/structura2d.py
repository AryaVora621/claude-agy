"""
Structura 2D: Standalone Desktop Finite Element Analysis (FEA) Studio.
Interactive 2D structural mechanics, continuum elasticity, stress contours,
deformation animation, and modal vibration analysis in Python Tkinter.
Zero external dependencies (Standard Library Only).
"""

import math
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Tuple, List

try:
    from .fea_engine import FEAModel, Node2D, TrussElement2D, CSTElement2D, Quad4Element2D
    from .presets import PRESETS
except ImportError:
    from fea_engine import FEAModel, Node2D, TrussElement2D, CSTElement2D, Quad4Element2D
    from presets import PRESETS


class Structura2DApp:
    """Main desktop graphical interface for Structura 2D FEA Studio."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Structura 2D - Finite Element Analysis & Continuum Mechanics Studio")
        self.root.geometry("1260x820")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#080C16")

        self.model: FEAModel = PRESETS["warren_truss"]()
        self.selected_preset = "warren_truss"

        # Viewport transform (World to Canvas)
        self.pan_x = 180.0
        self.pan_y = 480.0
        self.zoom = 55.0  # pixels per meter
        self.drag_start = (0, 0)

        # Visualization states
        self.render_mode = tk.StringVar(value="von_mises")  # 'von_mises', 'disp', 'sig_xx', 'sig_yy', 'tau_xy', 'wireframe'
        self.disp_scale = tk.DoubleVar(value=200.0)
        self.show_undeformed = tk.BooleanVar(value=True)
        self.show_nodes = tk.BooleanVar(value=True)
        self.show_loads = tk.BooleanVar(value=True)
        self.show_supports = tk.BooleanVar(value=True)
        self.show_reactions = tk.BooleanVar(value=True)

        # Modal vibration animation state
        self.animating_vibration = False
        self.vibration_phase = 0.0
        self.vibration_speed = 0.15

        # Selected node / element for inspector
        self.selected_node_id: Optional[int] = None
        self.selected_elem_info: Optional[str] = None

        self._build_ui()
        self.solve_and_redraw()

    def _build_ui(self):
        # 1. Top Header
        header = tk.Frame(self.root, bg="#0F172A", height=50, bd=0, highlightthickness=1, highlightbackground="#1E293B")
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header, text="STRUCTURA 2D", font=("Segoe UI", 13, "bold"),
            fg="#00F0FF", bg="#0F172A", padx=16
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            header, text="|  Linear Elastic Continuum FEA & Truss Mechanics",
            font=("Segoe UI", 9), fg="#94A3B8", bg="#0F172A"
        )
        sub_lbl.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            header, text="Ready", font=("Consolas", 9, "bold"),
            fg="#00FF9D", bg="#0F172A", padx=20
        )
        self.lbl_status.pack(side=tk.RIGHT)

        # 2. Main Workspace Layout: Left Sidebar, Canvas, Right Telemetry
        workspace = tk.Frame(self.root, bg="#080C16")
        workspace.pack(fill=tk.BOTH, expand=True)

        # --- Left Control Sidebar ---
        left_panel = tk.Frame(workspace, bg="#0D1322", width=280, highlightthickness=1, highlightbackground="#1E293B")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left_panel.pack_propagate(False)

        # Scrollable container for left panel
        l_canvas = tk.Canvas(left_panel, bg="#0D1322", bd=0, highlightthickness=0)
        l_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=l_canvas.yview)
        l_scrollable = tk.Frame(l_canvas, bg="#0D1322")
        l_scrollable.bind("<Configure>", lambda e: l_canvas.configure(scrollregion=l_canvas.bbox("all")))
        l_canvas.create_window((0, 0), window=l_scrollable, anchor="nw", width=265)
        l_canvas.configure(yview_command=l_scrollbar.set)
        l_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        l_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Preset Selection Group
        self._create_section_label(l_scrollable, "STRUCTURAL PRESETS")
        preset_frame = tk.Frame(l_scrollable, bg="#0D1322")
        preset_frame.pack(fill=tk.X, padx=8, pady=4)

        presets = [
            ("Warren Truss Bridge", "warren_truss"),
            ("Cantilever Beam (CST)", "cantilever_cst"),
            ("Kirsch Plate with Hole", "kirsch_plate_hole"),
            ("L-Shaped Bracket", "l_bracket"),
            ("Thick Pressure Cylinder", "thick_cylinder"),
            ("Quad4 Shear Wall", "quad4_shear_wall")
        ]

        self.preset_btns = {}
        for text, key in presets:
            btn = tk.Button(
                preset_frame, text=text, font=("Segoe UI", 8, "bold"),
                bg="#162035", fg="#CBD5E1", activebackground="#00F0FF", activeforeground="#000",
                bd=0, padx=8, pady=5, anchor="w",
                command=lambda k=key: self.load_preset(k)
            )
            btn.pack(fill=tk.X, pady=2)
            self.preset_btns[key] = btn

        # Render Field Mode
        self._create_section_label(l_scrollable, "CONTOUR RENDER MODE")
        render_frame = tk.Frame(l_scrollable, bg="#0D1322")
        render_frame.pack(fill=tk.X, padx=8, pady=4)

        modes = [
            ("Von Mises Stress (MPa)", "von_mises"),
            ("Displacement Mag (mm)", "disp"),
            ("Normal Stress σ_xx", "sig_xx"),
            ("Normal Stress σ_yy", "sig_yy"),
            ("Shear Stress τ_xy", "tau_xy"),
            ("Wireframe Grid", "wireframe")
        ]
        for text, val in modes:
            rb = tk.Radiobutton(
                render_frame, text=text, variable=self.render_mode, value=val,
                font=("Segoe UI", 8), fg="#94A3B8", selectcolor="#080C16",
                bg="#0D1322", activebackground="#0D1322", activeforeground="#00F0FF",
                command=self.redraw_canvas
            )
            rb.pack(anchor="w", pady=1)

        # Display Toggles
        self._create_section_label(l_scrollable, "DISPLAY OVERLAYS")
        chk_frame = tk.Frame(l_scrollable, bg="#0D1322")
        chk_frame.pack(fill=tk.X, padx=8, pady=4)

        toggles = [
            ("Show Undeformed Wireframe", self.show_undeformed),
            ("Show Nodal Points", self.show_nodes),
            ("Show External Load Arrows", self.show_loads),
            ("Show Support Constraints", self.show_supports),
            ("Show Reaction Vectors", self.show_reactions)
        ]
        for text, var in toggles:
            cb = tk.Checkbutton(
                chk_frame, text=text, variable=var,
                font=("Segoe UI", 8), fg="#94A3B8", selectcolor="#080C16",
                bg="#0D1322", activebackground="#0D1322", activeforeground="#00F0FF",
                command=self.redraw_canvas
            )
            cb.pack(anchor="w", pady=1)

        # Deformation Scale Slider
        self._create_section_label(l_scrollable, "DEFORMATION SCALE")
        scale_frame = tk.Frame(l_scrollable, bg="#0D1322")
        scale_frame.pack(fill=tk.X, padx=8, pady=4)

        self.scale_lbl = tk.Label(scale_frame, text="Scale: 200x", font=("Consolas", 8), fg="#00F0FF", bg="#0D1322")
        self.scale_lbl.pack(anchor="e")

        scale_slider = tk.Scale(
            scale_frame, from_=1.0, to_=1000.0, orient=tk.HORIZONTAL,
            variable=self.disp_scale, showvalue=False, bg="#162035", fg="#00F0FF",
            highlightthickness=0, bd=0, activebackground="#00F0FF",
            command=self._on_scale_change
        )
        scale_slider.pack(fill=tk.X, pady=2)

        # Modal Vibration Controls
        self._create_section_label(l_scrollable, "MODAL HARMONICS")
        vib_frame = tk.Frame(l_scrollable, bg="#0D1322")
        vib_frame.pack(fill=tk.X, padx=8, pady=4)

        self.btn_vib = tk.Button(
            vib_frame, text="Animate Vibration (Mode 1)", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#00FF9D", activebackground="#00FF9D", activeforeground="#000",
            bd=0, padx=6, pady=6, command=self.toggle_vibration
        )
        self.btn_vib.pack(fill=tk.X, pady=2)

        # Action Buttons
        act_frame = tk.Frame(l_scrollable, bg="#0D1322")
        act_frame.pack(fill=tk.X, padx=8, pady=12)

        btn_solve = tk.Button(
            act_frame, text="SOLVE FEA EQUILIBRIUM", font=("Segoe UI", 9, "bold"),
            bg="#00F0FF", fg="#080C16", activebackground="#33F4FF",
            bd=0, padx=8, pady=8, command=self.solve_and_redraw
        )
        btn_solve.pack(fill=tk.X, pady=3)

        btn_reset_view = tk.Button(
            act_frame, text="Reset Camera View", font=("Segoe UI", 8),
            bg="#162035", fg="#94A3B8", activebackground="#1E293B",
            bd=0, padx=8, pady=4, command=self.reset_view
        )
        btn_reset_view.pack(fill=tk.X, pady=2)

        # --- Center Drawing Canvas ---
        center_frame = tk.Frame(workspace, bg="#050811")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center_frame, bg="#050811", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas Mouse Bindings for Pan, Zoom, and Picking
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_zoom_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.15, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.85, e.x, e.y))

        # --- Right Telemetry & Inspector Sidebar ---
        right_panel = tk.Frame(workspace, bg="#0D1322", width=290, highlightthickness=1, highlightbackground="#1E293B")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)
        right_panel.pack_propagate(False)

        self._create_section_label(right_panel, "STRUCTURAL TELEMETRY")
        telem_box = tk.Frame(right_panel, bg="#0D1322")
        telem_box.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_telem_dofs = self._create_telemetry_row(telem_box, "Active DOFs:", "--")
        self.lbl_telem_disp = self._create_telemetry_row(telem_box, "Max Deflection:", "-- mm")
        self.lbl_telem_stress = self._create_telemetry_row(telem_box, "Peak Von Mises:", "-- MPa")
        self.lbl_telem_energy = self._create_telemetry_row(telem_box, "Strain Energy:", "-- J")
        self.lbl_telem_sf = self._create_telemetry_row(telem_box, "Yield Safety Factor:", "--")
        self.lbl_telem_freq = self._create_telemetry_row(telem_box, "Natural Freq f_1:", "-- Hz")

        # Stress Color Legend
        self._create_section_label(right_panel, "STRESS COLOR CONTOUR")
        legend_frame = tk.Frame(right_panel, bg="#0D1322")
        legend_frame.pack(fill=tk.X, padx=12, pady=4)

        self.legend_canvas = tk.Canvas(legend_frame, bg="#0D1322", height=24, bd=0, highlightthickness=0)
        self.legend_canvas.pack(fill=tk.X, pady=4)

        legend_labels = tk.Frame(legend_frame, bg="#0D1322")
        legend_labels.pack(fill=tk.X)
        self.lbl_legend_min = tk.Label(legend_labels, text="0 MPa", font=("Consolas", 7), fg="#94A3B8", bg="#0D1322")
        self.lbl_legend_min.pack(side=tk.LEFT)
        self.lbl_legend_max = tk.Label(legend_labels, text="-- MPa", font=("Consolas", 7), fg="#00F0FF", bg="#0D1322")
        self.lbl_legend_max.pack(side=tk.RIGHT)

        # Inspector Panel
        self._create_section_label(right_panel, "INSPECTOR / PROBE")
        insp_frame = tk.Frame(right_panel, bg="#090E1A", bd=1, relief=tk.SOLID)
        insp_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.txt_inspector = tk.Text(
            insp_frame, bg="#090E1A", fg="#E2E8F0", font=("Consolas", 8),
            bd=0, highlightthickness=0, wrap=tk.WORD
        )
        self.txt_inspector.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.txt_inspector.insert(tk.END, "Click any node or element on the canvas to inspect coordinates, local displacement, reaction forces, and stress components.")

    def _create_section_label(self, parent: tk.Widget, text: str):
        lbl = tk.Label(
            parent, text=text, font=("Segoe UI", 7, "bold"),
            fg="#00F0FF", bg="#0D1322", padx=8, pady=6, anchor="w"
        )
        lbl.pack(fill=tk.X, pady=(6, 0))

    def _create_telemetry_row(self, parent: tk.Widget, label_text: str, default_val: str) -> tk.Label:
        row = tk.Frame(parent, bg="#131C30", bd=0, padx=6, pady=4)
        row.pack(fill=tk.X, pady=2)
        lbl_k = tk.Label(row, text=label_text, font=("Segoe UI", 8), fg="#94A3B8", bg="#131C30")
        lbl_k.pack(side=tk.LEFT)
        lbl_v = tk.Label(row, text=default_val, font=("Consolas", 8, "bold"), fg="#00FF9D", bg="#131C30")
        lbl_v.pack(side=tk.RIGHT)
        return lbl_v

    def _draw_legend_bar(self, max_val_mpa: float):
        self.legend_canvas.delete("all")
        w = self.legend_canvas.winfo_width()
        if w < 50:
            w = 260
        h = 16

        for x in range(w):
            t = x / max(1, w - 1)
            col = self.get_contour_color(t)
            self.legend_canvas.create_line(x, 0, x, h, fill=col)

        self.lbl_legend_max.config(text=f"{max_val_mpa:.2f} MPa")

    def _on_scale_change(self, val):
        scale = float(val)
        self.scale_lbl.config(text=f"Scale: {scale:.0f}x")
        self.redraw_canvas()

    def load_preset(self, key: str):
        if key in PRESETS:
            self.selected_preset = key
            self.model = PRESETS[key]()
            # Update button styling
            for k, btn in self.preset_btns.items():
                if k == key:
                    btn.config(bg="#00F0FF", fg="#000")
                else:
                    btn.config(bg="#162035", fg="#CBD5E1")
            self.reset_view()
            self.solve_and_redraw()

    def reset_view(self):
        # Auto-fit model into canvas view
        if not self.model.nodes:
            return
        xs = [n.x for n in self.model.nodes.values()]
        ys = [n.y for n in self.model.nodes.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(0.2, max_x - min_x)
        span_y = max(0.2, max_y - min_y)

        cw = self.canvas.winfo_width() or 700
        ch = self.canvas.winfo_height() or 600

        self.zoom = min(cw * 0.7 / span_x, ch * 0.7 / span_y)
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0
        self.pan_x = cw / 2.0 - mid_x * self.zoom
        self.pan_y = ch / 2.0 + mid_y * self.zoom
        self.redraw_canvas()

    def solve_and_redraw(self):
        try:
            res = self.model.solve_static()
            self.lbl_status.config(text="Solved (Static Equilibrium)", fg="#00FF9D")

            # Update Telemetry Display
            self.lbl_telem_dofs.config(text=str(self.model.num_dofs))
            self.lbl_telem_disp.config(text=f"{res['max_disp'] * 1000.0:.3f} mm")
            self.lbl_telem_stress.config(text=f"{res['max_von_mises'] / 1e6:.2f} MPa")
            self.lbl_telem_energy.config(text=f"{res['strain_energy']:.3e} J")
            sf_text = f"{res['safety_factor']:.2f}" if res['safety_factor'] < 100 else "> 100"
            self.lbl_telem_sf.config(text=sf_text)
            self.lbl_telem_freq.config(text=f"{res['natural_frequency_hz']:.1f} Hz")

            self._draw_legend_bar(res['max_von_mises'] / 1e6)
            self.redraw_canvas()

        except Exception as e:
            self.lbl_status.config(text="Solver Error", fg="#FF3366")
            messagebox.showerror("FEA Solver Error", str(e))

    def _world_to_screen(self, wx: float, wy: float, def_x: float = 0.0, def_y: float = 0.0) -> Tuple[float, float]:
        scale = self.disp_scale.get()
        if self.animating_vibration:
            scale *= math.sin(self.vibration_phase)
        sx = self.pan_x + (wx + def_x * scale) * self.zoom
        sy = self.pan_y - (wy + def_y * scale) * self.zoom
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        wx = (sx - self.pan_x) / self.zoom
        wy = -(sy - self.pan_y) / self.zoom
        return wx, wy

    def get_contour_color(self, t: float) -> str:
        """Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)."""
        t = max(0.0, min(1.0, t))
        if t < 0.25:
            fac = t / 0.25
            r = 0
            g = int(fac * 240)
            b = 255
        elif t < 0.50:
            fac = (t - 0.25) / 0.25
            r = 0
            g = 255
            b = int((1.0 - fac) * 255)
        elif t < 0.75:
            fac = (t - 0.50) / 0.25
            r = int(fac * 255)
            g = 255
            b = 0
        else:
            fac = (t - 0.75) / 0.25
            r = 255
            g = int((1.0 - fac) * 200)
            b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    def redraw_canvas(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # 1. Draw Subtle Datum Coordinate Grid
        grid_step = 1.0  # meter
        if self.zoom < 20:
            grid_step = 2.0
        elif self.zoom > 100:
            grid_step = 0.5

        wx_min, wy_max = self._screen_to_world(0, 0)
        wx_max, wy_min = self._screen_to_world(cw, ch)

        x_start = math.floor(wx_min / grid_step) * grid_step
        y_start = math.floor(wy_min / grid_step) * grid_step

        x = x_start
        while x <= wx_max:
            sx, _ = self._world_to_screen(x, 0)
            self.canvas.create_line(sx, 0, sx, ch, fill="#0F172A", width=1)
            x += grid_step

        y = y_start
        while y <= wy_max:
            _, sy = self._world_to_screen(0, y)
            self.canvas.create_line(0, sy, cw, sy, fill="#0F172A", width=1)
            y += grid_step

        # Origin Axes Lines
        ox, oy = self._world_to_screen(0, 0)
        self.canvas.create_line(ox, 0, ox, ch, fill="#1E293B", width=1)
        self.canvas.create_line(0, oy, cw, oy, fill="#1E293B", width=1)

        mode = self.render_mode.get()
        max_vm = max(1e-6, self.model.max_von_mises)
        max_d = max(1e-9, self.model.max_disp)

        # 2. Draw Undeformed Mesh Wireframe (if enabled)
        if self.show_undeformed.get() and mode != "wireframe":
            # Undeformed CST
            for cst in self.model.cst_elements:
                n1 = self.model.nodes[cst.n1]
                n2 = self.model.nodes[cst.n2]
                n3 = self.model.nodes[cst.n3]
                p1 = self._world_to_screen(n1.x, n1.y)
                p2 = self._world_to_screen(n2.x, n2.y)
                p3 = self._world_to_screen(n3.x, n3.y)
                self.canvas.create_polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], fill="", outline="#1E293B", width=1)

            # Undeformed Quad4
            for q in self.model.quad_elements:
                n1, n2, n3, n4 = self.model.nodes[q.n1], self.model.nodes[q.n2], self.model.nodes[q.n3], self.model.nodes[q.n4]
                p1 = self._world_to_screen(n1.x, n1.y)
                p2 = self._world_to_screen(n2.x, n2.y)
                p3 = self._world_to_screen(n3.x, n3.y)
                p4 = self._world_to_screen(n4.x, n4.y)
                self.canvas.create_polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], fill="", outline="#1E293B", width=1)

            # Undeformed Truss
            for t in self.model.trusses:
                n1, n2 = self.model.nodes[t.n1], self.model.nodes[t.n2]
                p1 = self._world_to_screen(n1.x, n1.y)
                p2 = self._world_to_screen(n2.x, n2.y)
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#1E293B", width=1, dash=(3, 3))

        # 3. Draw Deformed Continuum Elements (CST & Quad4)
        for cst in self.model.cst_elements:
            n1 = self.model.nodes[cst.n1]
            n2 = self.model.nodes[cst.n2]
            n3 = self.model.nodes[cst.n3]
            p1 = self._world_to_screen(n1.x, n1.y, n1.ux, n1.uy)
            p2 = self._world_to_screen(n2.x, n2.y, n2.ux, n2.uy)
            p3 = self._world_to_screen(n3.x, n3.y, n3.ux, n3.uy)

            # Determine color
            col = "#1E293B"
            if mode == "von_mises":
                col = self.get_contour_color(cst.sigma_vM / max_vm)
            elif mode == "disp":
                avg_d = (math.hypot(n1.ux, n1.uy) + math.hypot(n2.ux, n2.uy) + math.hypot(n3.ux, n3.uy)) / 3.0
                col = self.get_contour_color(avg_d / max_d)
            elif mode == "sig_xx":
                norm = (cst.sigma_xx / max_vm + 1.0) * 0.5
                col = self.get_contour_color(norm)
            elif mode == "sig_yy":
                norm = (cst.sigma_yy / max_vm + 1.0) * 0.5
                col = self.get_contour_color(norm)
            elif mode == "tau_xy":
                norm = (abs(cst.tau_xy) / (max_vm * 0.6))
                col = self.get_contour_color(norm)
            elif mode == "wireframe":
                col = ""

            self.canvas.create_polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], fill=col, outline="#0F172A", width=1)

        for q in self.model.quad_elements:
            n1, n2, n3, n4 = self.model.nodes[q.n1], self.model.nodes[q.n2], self.model.nodes[q.n3], self.model.nodes[q.n4]
            p1 = self._world_to_screen(n1.x, n1.y, n1.ux, n1.uy)
            p2 = self._world_to_screen(n2.x, n2.y, n2.ux, n2.uy)
            p3 = self._world_to_screen(n3.x, n3.y, n3.ux, n3.uy)
            p4 = self._world_to_screen(n4.x, n4.y, n4.ux, n4.uy)

            col = "#1E293B"
            if mode == "von_mises":
                col = self.get_contour_color(q.sigma_vM / max_vm)
            elif mode == "disp":
                avg_d = (math.hypot(n1.ux, n1.uy) + math.hypot(n2.ux, n2.uy) + math.hypot(n3.ux, n3.uy) + math.hypot(n4.ux, n4.uy)) / 4.0
                col = self.get_contour_color(avg_d / max_d)
            elif mode == "sig_xx":
                norm = (q.sigma_xx / max_vm + 1.0) * 0.5
                col = self.get_contour_color(norm)
            elif mode == "sig_yy":
                norm = (q.sigma_yy / max_vm + 1.0) * 0.5
                col = self.get_contour_color(norm)
            elif mode == "tau_xy":
                norm = (abs(q.tau_xy) / (max_vm * 0.6))
                col = self.get_contour_color(norm)
            elif mode == "wireframe":
                col = ""

            self.canvas.create_polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], fill=col, outline="#0F172A", width=1)

        # 4. Draw Truss Elements (color coded: Blue = Tension, Red = Compression)
        for t in self.model.trusses:
            n1, n2 = self.model.nodes[t.n1], self.model.nodes[t.n2]
            p1 = self._world_to_screen(n1.x, n1.y, n1.ux, n1.uy)
            p2 = self._world_to_screen(n2.x, n2.y, n2.ux, n2.uy)

            # Color by tension/compression
            if mode == "von_mises":
                col = self.get_contour_color(abs(t.axial_stress) / max_vm)
            elif mode == "disp":
                avg_d = (math.hypot(n1.ux, n1.uy) + math.hypot(n2.ux, n2.uy)) / 2.0
                col = self.get_contour_color(avg_d / max_d)
            elif t.axial_force > 1.0:
                col = "#00F0FF"  # Tension
            elif t.axial_force < -1.0:
                col = "#FF3366"  # Compression
            else:
                col = "#94A3B8"

            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=col, width=3.5, capstyle=tk.ROUND)

        # 5. Draw Support Constraints (Triangle = Pin, Circle = Roller, Clamped Hatch)
        if self.show_supports.get():
            for node in self.model.nodes.values():
                if node.fix_x or node.fix_y:
                    px, py = self._world_to_screen(node.x, node.y, node.ux, node.uy)

                    if node.fix_x and node.fix_y:
                        # Fixed pin support: draw triangle
                        self.canvas.create_polygon(px, py, px - 8, py + 12, px + 8, py + 12, fill="#F59E0B", outline="#000")
                        self.canvas.create_line(px - 10, py + 13, px + 10, py + 13, fill="#F59E0B", width=2)
                    elif node.fix_y:
                        # Roller support: triangle + ground roller circles
                        self.canvas.create_polygon(px, py, px - 7, py + 9, px + 7, py + 9, fill="#38BDF8", outline="#000")
                        self.canvas.create_oval(px - 6, py + 10, px - 2, py + 14, fill="#FFF", outline="#38BDF8")
                        self.canvas.create_oval(px + 2, py + 10, px + 6, py + 14, fill="#FFF", outline="#38BDF8")
                    elif node.fix_x:
                        # Vertical roller: triangle oriented horizontally
                        self.canvas.create_polygon(px, py, px - 9, py - 7, px - 9, py + 7, fill="#38BDF8", outline="#000")

        # 6. Draw External Applied Load Arrows (Red arrows)
        if self.show_loads.get():
            for node in self.model.nodes.values():
                if abs(node.fx) > 1e-3 or abs(node.fy) > 1e-3:
                    px, py = self._world_to_screen(node.x, node.y, node.ux, node.uy)
                    f_mag = math.hypot(node.fx, node.fy)
                    arrow_len = min(60.0, max(22.0, math.sqrt(f_mag) * 0.12))
                    angle = math.atan2(-node.fy, node.fx)

                    ax = px + arrow_len * math.cos(angle)
                    ay = py + arrow_len * math.sin(angle)
                    self.canvas.create_line(ax, ay, px, py, arrow=tk.LAST, fill="#FF0055", width=2.5, arrowshape=(10, 12, 5))

        # 7. Draw Reaction Force Vectors (Green arrows at supports)
        if self.show_reactions.get():
            for node in self.model.nodes.values():
                if (node.fix_x or node.fix_y) and (abs(node.rx) > 10.0 or abs(node.ry) > 10.0):
                    px, py = self._world_to_screen(node.x, node.y, node.ux, node.uy)
                    r_mag = math.hypot(node.rx, node.ry)
                    arrow_len = min(50.0, max(18.0, math.sqrt(r_mag) * 0.10))
                    angle = math.atan2(-node.ry, node.rx)

                    ax = px + arrow_len * math.cos(angle)
                    ay = py + arrow_len * math.sin(angle)
                    self.canvas.create_line(px, py, ax, ay, arrow=tk.LAST, fill="#00FF9D", width=2.0, arrowshape=(8, 10, 4))

        # 8. Draw Nodal Circles & Selection Highlight
        if self.show_nodes.get():
            for nid, node in self.model.nodes.items():
                px, py = self._world_to_screen(node.x, node.y, node.ux, node.uy)
                is_selected = (nid == self.selected_node_id)
                rad = 4.5 if is_selected else 2.5
                col = "#00F0FF" if is_selected else "#64748B"
                out = "#FFF" if is_selected else "#0F172A"
                self.canvas.create_oval(px - rad, py - rad, px + rad, py + rad, fill=col, outline=out, width=1.5)

    def _on_canvas_click(self, event):
        # Find closest node to click
        click_sx, click_sy = event.x, event.y
        best_dist = 18.0  # Pixel snap threshold
        best_node = None

        for nid, node in self.model.nodes.items():
            px, py = self._world_to_screen(node.x, node.y, node.ux, node.uy)
            dist = math.hypot(click_sx - px, click_sy - py)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        if best_node:
            self.selected_node_id = best_node.id
            self._update_inspector_node(best_node)
        else:
            self.selected_node_id = None
            self.txt_inspector.delete("1.0", tk.END)
            self.txt_inspector.insert(tk.END, "No node selected.\n\nClick any node to inspect local displacements, forces, and constraints.")

        self.redraw_canvas()

    def _update_inspector_node(self, node: Node2D):
        self.txt_inspector.delete("1.0", tk.END)
        info = [
            f"=== NODE ID: {node.id} ===",
            f"Coordinates: ({node.x:.3f}, {node.y:.3f}) m",
            "",
            "--- DISPLACEMENTS ---",
            f"u_x : {node.ux * 1000.0:+.4f} mm",
            f"u_y : {node.uy * 1000.0:+.4f} mm",
            f"Mag : {math.hypot(node.ux, node.uy) * 1000.0:.4f} mm",
            "",
            "--- APPLIED FORCES ---",
            f"F_x : {node.fx:+.1f} N",
            f"F_y : {node.fy:+.1f} N",
            "",
            "--- BOUNDARY REACTIONS ---",
            f"Constraint X : {'FIXED' if node.fix_x else 'Free'}",
            f"Constraint Y : {'FIXED' if node.fix_y else 'Free'}",
            f"Reaction R_x : {node.rx:+.1f} N",
            f"Reaction R_y : {node.ry:+.1f} N",
            f"Total R Mag  : {math.hypot(node.rx, node.ry):.1f} N"
        ]
        self.txt_inspector.insert(tk.END, "\n".join(info))

    def _on_pan_start(self, event):
        self.drag_start = (event.x, event.y)

    def _on_pan_move(self, event):
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start = (event.x, event.y)
        self.redraw_canvas()

    def _on_zoom_wheel(self, event):
        factor = 1.12 if event.delta > 0 else 0.88
        self._zoom(factor, event.x, event.y)

    def _zoom(self, factor: float, cx: float, cy: float):
        new_zoom = max(5.0, min(500.0, self.zoom * factor))
        wx, wy = self._screen_to_world(cx, cy)
        self.zoom = new_zoom
        self.pan_x = cx - wx * self.zoom
        self.pan_y = cy + wy * self.zoom
        self.redraw_canvas()

    def toggle_vibration(self):
        self.animating_vibration = not self.animating_vibration
        if self.animating_vibration:
            self.btn_vib.config(text="Stop Vibration", bg="#FF3366", fg="#FFF")
            self._vibration_loop()
        else:
            self.btn_vib.config(text="Animate Vibration (Mode 1)", bg="#1E293B", fg="#00FF9D")
            self.redraw_canvas()

    def _vibration_loop(self):
        if self.animating_vibration:
            self.vibration_phase += self.vibration_speed
            self.redraw_canvas()
            self.root.after(30, self._vibration_loop)


def launch():
    root = tk.Tk()
    app = Structura2DApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
