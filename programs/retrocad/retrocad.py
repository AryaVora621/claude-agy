"""RetroCAD Desktop Studio.

Standalone standard library Tkinter 3D wireframe and solid parametric CAD modeler.
Features real-time 3D orbit/pan/zoom viewport, multiple render modes (Phosphor CRT,
Blueprint, Hidden-Line, Flat-Shaded), CSG Booleans, parametric primitives,
mechanical presets, live volume/surface inspection, and STL/OBJ/DXF/SVG export.
Zero external dependencies.
"""

import sys
import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple

try:
    from .geom import (
        Vec3, Mat4, Mesh, Face, create_box, create_cylinder, create_sphere,
        create_cone, create_torus, csg_boolean, export_stl_ascii, export_obj,
        export_dxf_r12, export_svg_wireframe
    )
    from .presets import PRESET_BUILDERS
except ImportError:
    from geom import (
        Vec3, Mat4, Mesh, Face, create_box, create_cylinder, create_sphere,
        create_cone, create_torus, csg_boolean, export_stl_ascii, export_obj,
        export_dxf_r12, export_svg_wireframe
    )
    from presets import PRESET_BUILDERS


class RetroCADApp:
    """RetroCAD Standalone Desktop CAD Workstation GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RetroCAD - Parametric 3D CAD Modeler and CSG Studio")
        self.root.geometry("1180x760")
        self.root.minsize(960, 640)
        self.root.configure(bg="#030712")

        # 3D Camera View State
        self.cam_yaw: float = 0.65       # radians azimuth
        self.cam_pitch: float = 0.42     # radians elevation
        self.cam_zoom: float = 3.8       # zoom scale factor
        self.cam_pan_x: float = 0.0      # screen pan X
        self.cam_pan_y: float = 0.0      # screen pan Y
        self.render_mode: str = "Flat Shaded"  # Phosphor CRT, Blueprint, Hidden Line, Flat Shaded

        # Mouse interaction tracking
        self.mouse_last_x: int = 0
        self.mouse_last_y: int = 0
        self.is_orbiting: bool = False
        self.is_panning: bool = False

        # Active Mesh Model
        self.active_mesh: Mesh = PRESET_BUILDERS["Bearing Housing"]()

        # Build GUI Components
        self._build_theme()
        self._build_menu()
        self._build_layout()

        # Canvas redraw bindings
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down_left)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up_left)

        self.canvas.bind("<ButtonPress-2>", self._on_mouse_down_right)
        self.canvas.bind("<B2-Motion>", self._on_mouse_drag_right)
        self.canvas.bind("<ButtonRelease-2>", self._on_mouse_up_right)

        self.canvas.bind("<ButtonPress-3>", self._on_mouse_down_right)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag_right)
        self.canvas.bind("<ButtonRelease-3>", self._on_mouse_up_right)

        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.9))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.redraw()

    def _build_theme(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0b0f19")
        style.configure("TLabel", background="#0b0f19", foreground="#f8fafc", font=("Segoe UI", 9))
        style.configure("TLabelframe", background="#0b0f19", foreground="#00f0ff")
        style.configure("TLabelframe.Label", background="#0b0f19", foreground="#00f0ff", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", background="#1e293b", foreground="#f8fafc", font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#334155")])

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root, bg="#0b0f19", fg="#f8fafc", activebackground="#1e293b", activeforeground="#00f0ff")

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0, bg="#0b0f19", fg="#f8fafc", activebackground="#1e293b")
        file_menu.add_command(label="Export STL (ASCII)...", command=self.export_stl)
        file_menu.add_command(label="Export Wavefront OBJ...", command=self.export_obj)
        file_menu.add_command(label="Export AutoCAD DXF...", command=self.export_dxf)
        file_menu.add_command(label="Export Vector SVG...", command=self.export_svg)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # Presets Menu
        preset_menu = tk.Menu(menubar, tearoff=0, bg="#0b0f19", fg="#f8fafc", activebackground="#1e293b")
        for name in PRESET_BUILDERS.keys():
            preset_menu.add_command(label=name, command=lambda n=name: self.load_preset(n))
        menubar.add_cascade(label="Presets", menu=preset_menu)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0, bg="#0b0f19", fg="#f8fafc", activebackground="#1e293b")
        view_menu.add_command(label="Isometric View", command=self.view_iso)
        view_menu.add_command(label="Top View (XY)", command=self.view_top)
        view_menu.add_command(label="Front View (XZ)", command=self.view_front)
        view_menu.add_command(label="Right View (YZ)", command=self.view_right)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Camera", command=self.reset_view)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        # Top Header Ribbon
        header = tk.Frame(self.root, bg="#0b0f19", height=42, relief="flat", bd=0)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(header, text="RETROCAD STUDIO", font=("Segoe UI", 11, "bold"), fg="#00f0ff", bg="#0b0f19")
        title_lbl.pack(side=tk.LEFT, padx=14, pady=8)

        ver_lbl = tk.Label(header, text="v2.4 First-Principles CAD Kernel", font=("Courier", 8), fg="#94a3b8", bg="#0b0f19")
        ver_lbl.pack(side=tk.LEFT, pady=8)

        # Quick Preset Buttons in Header
        for p_name in ["Bearing Housing", "Rocket Engine Nozzle", "Parametric Spur Gear"]:
            btn = tk.Button(
                header, text=p_name, font=("Segoe UI", 8, "bold"), bg="#1e293b", fg="#e2e8f0",
                activebackground="#334155", activeforeground="#00f0ff", bd=1, relief="solid",
                command=lambda n=p_name: self.load_preset(n)
            )
            btn.pack(side=tk.RIGHT, padx=4, pady=6)

        # Main Split Frame
        main_frame = tk.Frame(self.root, bg="#030712")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Sidebar (Tools & Properties)
        sidebar = tk.Frame(main_frame, bg="#0b0f19", width=280, relief="solid", bd=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)

        # 1. Shading Modes
        mode_box = ttk.LabelFrame(sidebar, text=" Render Style ")
        mode_box.pack(fill=tk.X, padx=10, pady=8)

        self.mode_var = tk.StringVar(value="Flat Shaded")
        for mode in ["Flat Shaded", "Phosphor CRT", "Blueprint", "Hidden Line"]:
            rb = tk.Radiobutton(
                mode_box, text=mode, value=mode, variable=self.mode_var,
                bg="#0b0f19", fg="#f8fafc", selectcolor="#1e293b",
                activebackground="#0b0f19", activeforeground="#00f0ff",
                command=self._on_mode_change
            )
            rb.pack(anchor=tk.W, padx=8, pady=2)

        # 2. Add Primitive Mesh
        prim_box = ttk.LabelFrame(sidebar, text=" Add Primitive ")
        prim_box.pack(fill=tk.X, padx=10, pady=8)

        prim_btn_frame = tk.Frame(prim_box, bg="#0b0f19")
        prim_btn_frame.pack(fill=tk.X, padx=6, pady=4)

        prims = [
            ("Box", lambda: self.add_primitive("box")),
            ("Cylinder", lambda: self.add_primitive("cylinder")),
            ("Sphere", lambda: self.add_primitive("sphere")),
            ("Cone", lambda: self.add_primitive("cone")),
            ("Torus", lambda: self.add_primitive("torus"))
        ]
        for idx, (label, cmd) in enumerate(prims):
            r = idx // 2
            c = idx % 2
            btn = tk.Button(
                prim_btn_frame, text=label, font=("Segoe UI", 8), width=12,
                bg="#1e293b", fg="#e2e8f0", bd=1, relief="solid",
                command=cmd
            )
            btn.grid(row=r, column=c, padx=3, pady=3)

        # 3. CSG Booleans
        csg_box = ttk.LabelFrame(sidebar, text=" CSG Booleans ")
        csg_box.pack(fill=tk.X, padx=10, pady=8)

        csg_btn_frame = tk.Frame(csg_box, bg="#0b0f19")
        csg_btn_frame.pack(fill=tk.X, padx=6, pady=4)

        btn_union = tk.Button(
            csg_btn_frame, text="Union (A + B)", font=("Segoe UI", 8), width=12,
            bg="#15803d", fg="#ffffff", bd=1, relief="solid",
            command=lambda: self.apply_csg("union")
        )
        btn_union.grid(row=0, column=0, padx=3, pady=3)

        btn_diff = tk.Button(
            csg_btn_frame, text="Subtract (A - B)", font=("Segoe UI", 8), width=12,
            bg="#b91c1c", fg="#ffffff", bd=1, relief="solid",
            command=lambda: self.apply_csg("difference")
        )
        btn_diff.grid(row=0, column=1, padx=3, pady=3)

        btn_inter = tk.Button(
            csg_btn_frame, text="Intersect (A & B)", font=("Segoe UI", 8), width=12,
            bg="#0369a1", fg="#ffffff", bd=1, relief="solid",
            command=lambda: self.apply_csg("intersection")
        )
        btn_inter.grid(row=1, column=0, columnspan=2, padx=3, pady=3, sticky="ew")

        # 4. Mechanical Presets
        presets_box = ttk.LabelFrame(sidebar, text=" Mechanical Presets ")
        presets_box.pack(fill=tk.X, padx=10, pady=8)

        self.preset_combo = ttk.Combobox(
            presets_box, values=list(PRESET_BUILDERS.keys()), state="readonly"
        )
        self.preset_combo.set("Bearing Housing")
        self.preset_combo.pack(fill=tk.X, padx=8, pady=4)
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset_combo.get()))

        # 5. Technical Inspection & HUD
        inspect_box = ttk.LabelFrame(sidebar, text=" Model Inspection ")
        inspect_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.info_text = tk.Text(
            inspect_box, bg="#020617", fg="#38bdf8", font=("Courier", 8),
            bd=0, padx=6, pady=6, height=10
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Center 3D Canvas
        center_frame = tk.Frame(main_frame, bg="#030712")
        center_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center_frame, bg="#030712", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Floating View Preset Controls
        view_ctrl = tk.Frame(center_frame, bg="#0b0f19", bd=1, relief="solid")
        view_ctrl.place(relx=0.02, rely=0.02)

        tk.Button(view_ctrl, text="ISO", font=("Segoe UI", 8, "bold"), bg="#1e293b", fg="#00f0ff", bd=0, command=self.view_iso).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(view_ctrl, text="TOP", font=("Segoe UI", 8), bg="#1e293b", fg="#e2e8f0", bd=0, command=self.view_top).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(view_ctrl, text="FRONT", font=("Segoe UI", 8), bg="#1e293b", fg="#e2e8f0", bd=0, command=self.view_front).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(view_ctrl, text="RIGHT", font=("Segoe UI", 8), bg="#1e293b", fg="#e2e8f0", bd=0, command=self.view_right).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(view_ctrl, text="RESET", font=("Segoe UI", 8), bg="#334155", fg="#fca5a5", bd=0, command=self.reset_view).pack(side=tk.LEFT, padx=3, pady=2)

        # Bottom Status Bar
        statusbar = tk.Frame(self.root, bg="#0b0f19", height=24)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_lbl = tk.Label(
            statusbar, text="Left-Drag: Orbit | Right-Drag: Pan | Scroll: Zoom | Presets or CAD Primitives",
            font=("Segoe UI", 8), fg="#94a3b8", bg="#0b0f19"
        )
        self.status_lbl.pack(side=tk.LEFT, padx=12)

    def _on_mode_change(self) -> None:
        self.render_mode = self.mode_var.get()
        self.redraw()

    # -------------------------------------------------------------
    # Mouse Orbit, Pan, and Zoom Controls
    # -------------------------------------------------------------
    def _on_mouse_down_left(self, event: tk.Event) -> None:
        self.is_orbiting = True
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_drag_left(self, event: tk.Event) -> None:
        if not self.is_orbiting:
            return
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.cam_yaw += dx * 0.012
        self.cam_pitch = max(-1.5, min(1.5, self.cam_pitch + dy * 0.012))
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y
        self.redraw()

    def _on_mouse_up_left(self, event: tk.Event) -> None:
        self.is_orbiting = False

    def _on_mouse_down_right(self, event: tk.Event) -> None:
        self.is_panning = True
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_drag_right(self, event: tk.Event) -> None:
        if not self.is_panning:
            return
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.cam_pan_x += dx
        self.cam_pan_y += dy
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y
        self.redraw()

    def _on_mouse_up_right(self, event: tk.Event) -> None:
        self.is_panning = False

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self._zoom(1.1)
        else:
            self._zoom(0.9)

    def _zoom(self, factor: float) -> None:
        self.cam_zoom = max(0.5, min(25.0, self.cam_zoom * factor))
        self.redraw()

    # -------------------------------------------------------------
    # Camera Viewpoint Presets
    # -------------------------------------------------------------
    def view_iso(self) -> None:
        self.cam_yaw = math.radians(45.0)
        self.cam_pitch = math.radians(30.0)
        self.redraw()

    def view_top(self) -> None:
        self.cam_yaw = 0.0
        self.cam_pitch = math.radians(89.0)
        self.redraw()

    def view_front(self) -> None:
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.redraw()

    def view_right(self) -> None:
        self.cam_yaw = math.radians(90.0)
        self.cam_pitch = 0.0
        self.redraw()

    def reset_view(self) -> None:
        self.cam_yaw = 0.65
        self.cam_pitch = 0.42
        self.cam_zoom = 3.8
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0
        self.redraw()

    # -------------------------------------------------------------
    # Primitive and Preset Management
    # -------------------------------------------------------------
    def load_preset(self, name: str) -> None:
        builder = PRESET_BUILDERS.get(name)
        if builder:
            self.active_mesh = builder()
            self.redraw()
            self._update_inspection()

    def add_primitive(self, kind: str) -> None:
        if kind == "box":
            self.active_mesh = create_box(40.0, 40.0, 40.0, "Box40")
        elif kind == "cylinder":
            self.active_mesh = create_cylinder(20.0, 50.0, 24, "Cyl20x50")
        elif kind == "sphere":
            self.active_mesh = create_sphere(25.0, 14, 20, "Sphere25")
        elif kind == "cone":
            self.active_mesh = create_cone(25.0, 50.0, 24, "Cone25x50")
        elif kind == "torus":
            self.active_mesh = create_torus(30.0, 8.0, 24, 14, "Torus30x8")
        self.redraw()
        self._update_inspection()

    def apply_csg(self, operation: str) -> None:
        # Perform CSG with an intersecting sphere cutter
        cutter = create_sphere(24.0, 12, 16, "CutterSphere")
        cutter.transform(Mat4.translation(10.0, 10.0, 10.0))
        self.active_mesh = csg_boolean(self.active_mesh, cutter, operation)
        self.redraw()
        self._update_inspection()

    def _update_inspection(self) -> None:
        min_pt, max_pt, center, size = self.active_mesh.compute_bounding_box()
        area = self.active_mesh.surface_area()
        vol = self.active_mesh.volume()

        text = (
            f"PART: {self.active_mesh.name}\n"
            f"---------------------------\n"
            f"Vertices : {len(self.active_mesh.vertices)}\n"
            f"Faces    : {len(self.active_mesh.faces)}\n"
            f"Edges    : {len(self.active_mesh.edges)}\n"
            f"---------------------------\n"
            f"BOUNDING BOX (mm):\n"
            f" X : {size.x:.2f}\n"
            f" Y : {size.y:.2f}\n"
            f" Z : {size.z:.2f}\n"
            f"Center: ({center.x:.1f}, {center.y:.1f}, {center.z:.1f})\n"
            f"---------------------------\n"
            f"Surface Area: {area:.1f} mm²\n"
            f"Volume      : {vol:.1f} mm³\n"
        )
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)

    # -------------------------------------------------------------
    # 3D Viewport Drawing & Projection Pipeline
    # -------------------------------------------------------------
    def redraw(self) -> None:
        cv = self.canvas
        cv.delete("all")

        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 10 or h < 10:
            return

        cx = w * 0.5 + self.cam_pan_x
        cy = h * 0.5 + self.cam_pan_y

        # Compute scaling based on model bounds
        _, _, center, size = self.active_mesh.compute_bounding_box()
        max_dim = max(size.x, size.y, size.z, 1.0)
        base_scale = (min(w, h) * 0.35) / max_dim
        scale = base_scale * (self.cam_zoom / 3.8)

        # 3D Rotation Matrix
        rot_y = Mat4.rotation_z(self.cam_yaw)
        rot_x = Mat4.rotation_x(self.cam_pitch)
        rot_mat = rot_x.mul(rot_y)

        # 1. Draw Ground Reference Grid
        self._draw_ground_grid(cv, rot_mat, cx, cy, scale)

        # 2. Project Vertices
        projected: List[Tuple[float, float, float]] = []
        for v in self.active_mesh.vertices:
            rel = v - center
            rot = rot_mat.transform_point(rel)
            px = cx + rot.x * scale
            py = cy - rot.y * scale
            projected.append((px, py, rot.z))

        # 3. Render According to Chosen Shading Mode
        if self.render_mode == "Flat Shaded":
            self._render_flat_shaded(cv, projected, rot_mat)
        elif self.render_mode == "Phosphor CRT":
            self._render_wireframe(cv, projected, wire_color="#22c55e", bg_color="#052e16")
        elif self.render_mode == "Blueprint":
            self._render_wireframe(cv, projected, wire_color="#00f0ff", bg_color="#082f49")
        elif self.render_mode == "Hidden Line":
            self._render_hidden_line(cv, projected, rot_mat)

        # 4. Draw 3D Coordinate Orientation Axis Gizmo (Bottom-Left)
        self._draw_axis_gizmo(cv, rot_mat, 60, h - 60)

        # 5. Draw Viewport Overlay Info
        cv.create_text(
            16, 20, text=f"MODEL: {self.active_mesh.name} | MODE: {self.render_mode} | ZOOM: {self.cam_zoom:.1f}x",
            fill="#94a3b8", font=("Segoe UI", 9, "bold"), anchor="nw"
        )

    def _draw_ground_grid(self, cv: tk.Canvas, rot_mat: Mat4, cx: float, cy: float, scale: float) -> None:
        """Draw ground datum grid in XY plane."""
        grid_size = 60.0
        grid_step = 15.0
        steps = int(grid_size / grid_step)

        grid_color = "#111827"
        axis_color = "#1f293d"

        for i in range(-steps, steps + 1):
            coord = i * grid_step
            # Line parallel to X
            p1 = rot_mat.transform_point(Vec3(-grid_size, coord, -30.0))
            p2 = rot_mat.transform_point(Vec3( grid_size, coord, -30.0))
            cv.create_line(cx + p1.x * scale, cy - p1.y * scale, cx + p2.x * scale, cy - p2.y * scale, fill=axis_color if i == 0 else grid_color)

            # Line parallel to Y
            p3 = rot_mat.transform_point(Vec3(coord, -grid_size, -30.0))
            p4 = rot_mat.transform_point(Vec3(coord,  grid_size, -30.0))
            cv.create_line(cx + p3.x * scale, cy - p3.y * scale, cx + p4.x * scale, cy - p4.y * scale, fill=axis_color if i == 0 else grid_color)

    def _render_flat_shaded(self, cv: tk.Canvas, projected: List[Tuple[float, float, float]], rot_mat: Mat4) -> None:
        """Painter algorithm depth-sorted flat polygon shading with directional lighting."""
        light_dir = Vec3(0.577, 0.577, 0.577).normalize()

        # Build depth-sorted face list
        face_list = []
        for face in self.active_mesh.faces:
            n = len(face.indices)
            if n < 3:
                continue
            # Calculate average depth Z
            avg_z = sum(projected[idx][2] for idx in face.indices) / n
            face_list.append((avg_z, face))

        # Sort back-to-front (smallest Z drawn first)
        face_list.sort(key=lambda item: item[0])

        for _, face in face_list:
            # Transform normal to view space
            norm_rot = rot_mat.transform_vector(face.normal).normalize()
            # Lambertian cosine law dot product
            diffuse = max(0.0, norm_rot.dot(light_dir))
            brightness = 0.25 + 0.75 * diffuse

            # Convert base color with brightness
            rgb = self._hex_to_rgb(face.color)
            shaded_rgb = (int(rgb[0] * brightness), int(rgb[1] * brightness), int(rgb[2] * brightness))
            hex_color = f"#{shaded_rgb[0]:02x}{shaded_rgb[1]:02x}{shaded_rgb[2]:02x}"

            poly_pts = []
            for idx in face.indices:
                poly_pts.extend([projected[idx][0], projected[idx][1]])

            cv.create_polygon(poly_pts, fill=hex_color, outline="#1e293b", width=1)

    def _render_wireframe(self, cv: tk.Canvas, projected: List[Tuple[float, float, float]], wire_color: str, bg_color: str) -> None:
        """Retro vector display glowing wireframe."""
        for i1, i2 in self.active_mesh.edges:
            p1 = projected[i1]
            p2 = projected[i2]
            # Draw subtle glow line behind
            cv.create_line(p1[0], p1[1], p2[0], p2[1], fill=bg_color, width=3)
            # Sharp vector wire on top
            cv.create_line(p1[0], p1[1], p2[0], p2[1], fill=wire_color, width=1.5)

    def _render_hidden_line(self, cv: tk.Canvas, projected: List[Tuple[float, float, float]], rot_mat: Mat4) -> None:
        """Back-face culled hidden line wireframe."""
        # Solid fill to occlude behind, crisp outline
        for face in self.active_mesh.faces:
            norm_rot = rot_mat.transform_vector(face.normal)
            if norm_rot.z > -0.05:  # Facing camera
                poly_pts = []
                for idx in face.indices:
                    poly_pts.extend([projected[idx][0], projected[idx][1]])
                cv.create_polygon(poly_pts, fill="#030712", outline="#38bdf8", width=1.5)

    def _draw_axis_gizmo(self, cv: tk.Canvas, rot_mat: Mat4, gx: float, gy: float) -> None:
        """Draw small XYZ orientation triad in bottom left corner."""
        axis_len = 32.0
        orig = Vec3(0, 0, 0)
        p_orig = rot_mat.transform_point(orig)

        axes = [
            (Vec3(axis_len, 0, 0), "#ef4444", "X"),
            (Vec3(0, axis_len, 0), "#10b981", "Y"),
            (Vec3(0, 0, axis_len), "#3b82f6", "Z")
        ]

        for pt, col, label in axes:
            rot_pt = rot_mat.transform_point(pt)
            x1 = gx + p_orig.x
            y1 = gy - p_orig.y
            x2 = gx + rot_pt.x
            y2 = gy - rot_pt.y
            cv.create_line(x1, y1, x2, y2, fill=col, width=2.5, arrow=tk.LAST)
            cv.create_text(x2 + 6, y2 - 2, text=label, fill=col, font=("Segoe UI", 8, "bold"))

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        h = hex_str.lstrip('#')
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        return (59, 130, 246)

    # -------------------------------------------------------------
    # CAD File Export Actions
    # -------------------------------------------------------------
    def export_stl(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".stl",
            filetypes=[("Stereolithography CAD", "*.stl"), ("All Files", "*.*")],
            title="Export ASCII STL Mesh"
        )
        if path:
            stl_data = export_stl_ascii(self.active_mesh)
            with open(path, "w") as f:
                f.write(stl_data)
            messagebox.showinfo("Export Successful", f"Saved STL model to:\n{path}")

    def export_obj(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All Files", "*.*")],
            title="Export Wavefront OBJ Mesh"
        )
        if path:
            obj_data = export_obj(self.active_mesh)
            with open(path, "w") as f:
                f.write(obj_data)
            messagebox.showinfo("Export Successful", f"Saved OBJ model to:\n{path}")

    def export_dxf(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("AutoCAD DXF R12", "*.dxf"), ("All Files", "*.*")],
            title="Export AutoCAD DXF"
        )
        if path:
            dxf_data = export_dxf_r12(self.active_mesh)
            with open(path, "w") as f:
                f.write(dxf_data)
            messagebox.showinfo("Export Successful", f"Saved DXF model to:\n{path}")

    def export_svg(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("Vector SVG", "*.svg"), ("All Files", "*.*")],
            title="Export Technical Vector SVG"
        )
        if path:
            svg_data = export_svg_wireframe(self.active_mesh, self.cam_yaw, self.cam_pitch)
            with open(path, "w") as f:
                f.write(svg_data)
            messagebox.showinfo("Export Successful", f"Saved SVG technical drawing to:\n{path}")

    def on_close(self) -> None:
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = RetroCADApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
