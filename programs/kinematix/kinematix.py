"""
KineMatix 3D: Standalone Desktop Robotics & Multibody Inverse Kinematics Studio.
Interactive 3D serial manipulators, DH parameters, DLS inverse kinematics,
Yoshikawa manipulability ellipsoids, and Stewart-Gough parallel hexapods.
Standard library Python tkinter: zero external dependencies.
"""

import sys
import os
import math
import time
from typing import List, Tuple, Optional, Dict, Any
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.kinematix.kinematics_engine import (
    Vector3,
    Matrix4,
    DHLink,
    SerialRobotArm,
    StewartPlatform,
)
from programs.kinematix.presets import PRESETS


class Camera3D:
    """Spherical orbital camera with 3D perspective projection and depth sorting."""

    def __init__(
        self,
        azimuth: float = 0.85,
        elevation: float = 0.45,
        distance: float = 2.4,
        center: Optional[Vector3] = None,
    ):
        self.azimuth = azimuth  # radians around Z axis
        self.elevation = elevation  # radians above XY plane
        self.distance = distance  # distance to focus center
        self.center = center if center is not None else Vector3(0.0, 0.0, 0.25)
        self.focal_length = 580.0

    def get_eye_position(self) -> Vector3:
        ce = math.cos(self.elevation)
        se = math.sin(self.elevation)
        ca = math.cos(self.azimuth)
        sa = math.sin(self.azimuth)

        x = self.center.x + self.distance * ce * sa
        y = self.center.y - self.distance * ce * ca
        z = self.center.z + self.distance * se
        return Vector3(x, y, z)

    def project_point(
        self, p: Vector3, screen_w: float, screen_h: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Projects 3D world point to 2D screen coordinates with perspective division.
        Returns (screen_x, screen_y, depth_z) or None if point is behind camera plane.
        """
        eye = self.get_eye_position()

        # Camera coordinate frame vectors
        forward = (self.center - eye).normalized()
        up_world = Vector3(0.0, 0.0, 1.0)
        right = forward.cross(up_world).normalized()
        if right.norm_sq() < 1e-6:
            right = Vector3(1.0, 0.0, 0.0)
        up = right.cross(forward).normalized()

        # Vector from camera eye to point
        rel = p - eye

        # Camera space coordinates
        x_cam = rel.dot(right)
        y_cam = rel.dot(up)
        z_cam = rel.dot(forward)

        if z_cam < 0.08:
            return None  # Behind near clipping plane

        # Perspective division
        inv_z = self.focal_length / z_cam
        sx = screen_w * 0.5 + x_cam * inv_z
        sy = screen_h * 0.5 - y_cam * inv_z
        return (sx, sy, z_cam)


class KineMatixApp:
    """Master Desktop GUI Application for KineMatix 3D Studio."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KineMatix 3D: Robotics & Multibody Kinematics Studio")
        self.root.geometry("1280x840")
        self.root.minsize(1080, 720)
        self.root.configure(bg="#070A13")

        # Camera and 3D viewport state
        self.camera = Camera3D()
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.interaction_mode = "orbit"  # "orbit" or "target_drag"

        # Active robot model state
        self.current_preset_key = "puma560"
        self.system: Any = None
        self.metadata: Dict[str, Any] = {}
        self.is_serial = True

        # Serial robot state
        self.joint_angles: List[float] = []
        self.target_pos = Vector3(0.35, 0.20, 0.35)
        self.auto_solve_ik = True
        self.show_ellipsoid = True
        self.show_trail = True
        self.ik_damping = 0.06
        self.ik_max_iter = 50
        self.ik_tolerance = 0.002

        # Parallel Stewart platform state
        self.plat_trans = Vector3(0.0, 0.0, 0.0)
        self.plat_rpy = [0.0, 0.0, 0.0]  # roll, pitch, yaw (radians)

        # Trajectory generation state
        self.traj_active = False
        self.traj_type = "circle"
        self.traj_time = 0.0
        self.traj_speed = 1.0
        self.trail_buffer: List[Vector3] = []
        self.max_trail_points = 180

        # Performance monitoring
        self.fps = 60.0
        self.last_frame_time = time.time()

        # UI element collections
        self.joint_sliders: List[tk.Scale] = []
        self.joint_readouts: List[tk.Label] = []
        self.target_sliders: Dict[str, tk.Scale] = {}
        self.target_labels: Dict[str, tk.Label] = {}

        self._load_preset(self.current_preset_key, initial=True)
        self._build_ui()
        self._bind_events()
        self._animate()

    def _load_preset(self, key: str, initial: bool = False):
        """Instantiate selected robotic preset model."""
        self.current_preset_key = key
        self.system, self.metadata = PRESETS[key]()
        self.is_serial = (self.metadata.get("type") == "serial")
        self.trail_buffer.clear()

        if self.is_serial:
            self.joint_angles = list(self.metadata.get("initial_q", [0.0] * self.system.num_joints))
            self.target_pos = self.metadata.get("default_target", Vector3(0.35, 0.20, 0.30))
            if self.auto_solve_ik and not initial:
                self._solve_ik_step()
        else:
            self.plat_trans = self.metadata.get("default_translation", Vector3(0.0, 0.0, 0.0))
            rpy = self.metadata.get("default_rpy", (0.0, 0.0, 0.0))
            self.plat_rpy = [rpy[0], rpy[1], rpy[2]]

    def _build_ui(self):
        """Construct dark theme desktop workspace and docking layout."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#070A13")
        style.configure("TLabel", background="#070A13", foreground="#F1F5F9", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"), foreground="#38BDF8")
        style.configure("Sub.TLabel", font=("Helvetica", 9), foreground="#94A3B8")
        style.configure("TButton", background="#111B33", foreground="#F1F5F9", borderwidth=1, font=("Helvetica", 9, "bold"))
        style.map("TButton", background=[("active", "#1D2D56"), ("pressed", "#38BDF8")])
        style.configure("TNotebook", background="#0B1120", borderwidth=0)
        style.configure("TNotebook.Tab", background="#111B33", foreground="#94A3B8", padding=[10, 4], font=("Helvetica", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#1D2D56")], foreground=[("selected", "#38BDF8")])

        # Top Control & Navigation Ribbon
        top_bar = tk.Frame(self.root, bg="#0B1120", height=50, padx=16, pady=8, highlightbackground="#1E293B", highlightthickness=1)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        lbl_logo = tk.Label(top_bar, text="KINEMATIX 3D", font=("Helvetica", 13, "bold"), fg="#38BDF8", bg="#0B1120")
        lbl_logo.pack(side=tk.LEFT, padx=(0, 6))

        lbl_tag = tk.Label(top_bar, text="Robotics & Multibody Inverse Kinematics", font=("Helvetica", 9), fg="#94A3B8", bg="#0B1120")
        lbl_tag.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(top_bar, text="Robot Preset:", font=("Helvetica", 9, "bold"), fg="#F1F5F9", bg="#0B1120").pack(side=tk.LEFT, padx=(0, 6))
        self.preset_combo = ttk.Combobox(
            top_bar,
            values=[
                "puma560: PUMA 560 6-DOF Industrial Arm",
                "ur5: UR5 Collaborative Cobot",
                "scara: SCARA 4-DOF Assembly Robot",
                "stanford: Stanford Arm 6-DOF (1969)",
                "humanoid7: 7-DOF Anthropomorphic Arm",
                "hexapod: Stewart-Gough Parallel Hexapod",
            ],
            state="readonly",
            width=36,
        )
        self.preset_combo.current(0)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)
        self.preset_combo.pack(side=tk.LEFT, padx=(0, 16))

        # Viewport control buttons
        btn_reset_cam = tk.Button(
            top_bar,
            text="Reset Camera",
            command=self._reset_camera,
            bg="#111B33",
            fg="#F1F5F9",
            activebackground="#1D2D56",
            activeforeground="#38BDF8",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            font=("Helvetica", 9, "bold"),
        )
        btn_reset_cam.pack(side=tk.LEFT, padx=(0, 8))

        btn_home_pose = tk.Button(
            top_bar,
            text="Home Pose",
            command=self._home_pose,
            bg="#111B33",
            fg="#F1F5F9",
            activebackground="#1D2D56",
            activeforeground="#38BDF8",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            font=("Helvetica", 9, "bold"),
        )
        btn_home_pose.pack(side=tk.LEFT, padx=(0, 16))

        # Mouse mode selector
        tk.Label(top_bar, text="Mouse Action:", font=("Helvetica", 9, "bold"), fg="#F1F5F9", bg="#0B1120").pack(side=tk.LEFT, padx=(0, 6))
        self.mouse_mode_var = tk.StringVar(value="orbit")
        rb_orbit = tk.Radiobutton(
            top_bar, text="Orbit View", variable=self.mouse_mode_var, value="orbit",
            bg="#0B1120", fg="#38BDF8", selectcolor="#070A13", activebackground="#0B1120",
            font=("Helvetica", 9), command=self._update_mouse_mode
        )
        rb_orbit.pack(side=tk.LEFT, padx=(0, 6))
        rb_target = tk.Radiobutton(
            top_bar, text="Drag 3D Target", variable=self.mouse_mode_var, value="target_drag",
            bg="#0B1120", fg="#F43F5E", selectcolor="#070A13", activebackground="#0B1120",
            font=("Helvetica", 9), command=self._update_mouse_mode
        )
        rb_target.pack(side=tk.LEFT)

        # Main horizontal split: 3D Viewport on Left, Control Dock on Right
        content_frame = tk.Frame(self.root, bg="#070A13")
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 3D Canvas
        self.canvas_frame = tk.Frame(content_frame, bg="#050810", highlightbackground="#1E293B", highlightthickness=1)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#050810", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Dock for Interactive Controls and Diagnostics
        self.dock_frame = tk.Frame(content_frame, bg="#0B1120", width=380, highlightbackground="#1E293B", highlightthickness=1)
        self.dock_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        self.dock_frame.pack_propagate(False)

        self._build_dock()

        # Bottom Telemetry Bar
        bottom_bar = tk.Frame(self.root, bg="#0B1120", height=32, padx=16, pady=4, highlightbackground="#1E293B", highlightthickness=1)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_telemetry = tk.Label(
            bottom_bar,
            text="End-Effector: X: 0.000 m  Y: 0.000 m  Z: 0.000 m | Target Dist: 0.0 mm | Manipulability w: 0.0000 | IK: Idle",
            font=("Courier", 9),
            fg="#38BDF8",
            bg="#0B1120",
        )
        self.lbl_telemetry.pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(bottom_bar, text="FPS: 60.0", font=("Courier", 9), fg="#10B981", bg="#0B1120")
        self.lbl_fps.pack(side=tk.RIGHT)

    def _build_dock(self):
        """Populate the right control dock with tabbed panels."""
        # Info header
        self.lbl_model_title = tk.Label(
            self.dock_frame, text=self.metadata.get("name", "Robot Manipulator"),
            font=("Helvetica", 11, "bold"), fg="#38BDF8", bg="#0B1120", wraplength=350, justify=tk.LEFT
        )
        self.lbl_model_title.pack(anchor="w", padx=12, pady=(10, 2))

        self.lbl_model_desc = tk.Label(
            self.dock_frame, text=self.metadata.get("description", ""),
            font=("Helvetica", 8), fg="#94A3B8", bg="#0B1120", wraplength=350, justify=tk.LEFT
        )
        self.lbl_model_desc.pack(anchor="w", padx=12, pady=(0, 10))

        # Notebook tabs
        self.notebook = ttk.Notebook(self.dock_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Tab 1: Joint Controls
        self.tab_joints = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_joints, text="Joint Angles")
        self._build_tab_joints()

        # Tab 2: Inverse Kinematics
        self.tab_ik = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ik, text="Inverse Kinematics")
        self._build_tab_ik()

        # Tab 3: Trajectory Generator
        self.tab_traj = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_traj, text="Trajectories")
        self._build_tab_traj()

    def _build_tab_joints(self):
        """Construct sliders and readouts for every robot joint."""
        for widget in self.tab_joints.winfo_children():
            widget.destroy()

        scroll_canvas = tk.Canvas(self.tab_joints, bg="#070A13", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_joints, orient="vertical", command=scroll_canvas.yview)
        self.joint_scroll_frame = tk.Frame(scroll_canvas, bg="#070A13")

        self.joint_scroll_frame.bind(
            "<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=self.joint_scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.joint_sliders: List[tk.Scale] = []
        self.joint_readouts: List[tk.Label] = []

        if self.is_serial:
            for i, link in enumerate(self.system.links):
                f = tk.Frame(self.joint_scroll_frame, bg="#0B1120", padx=8, pady=6, highlightbackground="#1E293B", highlightthickness=1)
                f.pack(fill=tk.X, expand=True, pady=4, padx=4)

                type_label = "Revolute" if link.is_revolute else "Prismatic"
                lbl_name = tk.Label(f, text=f"Joint {i + 1}: {link.name} ({type_label})", font=("Helvetica", 9, "bold"), fg="#F1F5F9", bg="#0B1120")
                lbl_name.pack(anchor="w")

                val_curr = self.joint_angles[i] if i < len(self.joint_angles) else 0.0

                if link.is_revolute:
                    deg_min = math.degrees(link.joint_min)
                    deg_max = math.degrees(link.joint_max)
                    deg_val = math.degrees(val_curr)

                    lbl_val = tk.Label(f, text=f"{deg_val:+.1f} deg ({val_curr:+.3f} rad)", font=("Courier", 8), fg="#38BDF8", bg="#0B1120")
                    lbl_val.pack(anchor="w")
                    self.joint_readouts.append(lbl_val)

                    slider = tk.Scale(
                        f, from_=deg_min, to=deg_max, resolution=0.5, orient=tk.HORIZONTAL,
                        bg="#0B1120", fg="#94A3B8", activebackground="#38BDF8",
                        troughcolor="#111B33", highlightthickness=0, showvalue=False,
                        command=lambda v, idx=i: self._on_joint_slider(idx, float(v))
                    )
                    slider.set(deg_val)
                    slider.pack(fill=tk.X, pady=2)
                    self.joint_sliders.append(slider)
                else:
                    # Prismatic
                    mm_min = link.joint_min * 1000.0
                    mm_max = link.joint_max * 1000.0
                    mm_val = val_curr * 1000.0

                    lbl_val = tk.Label(f, text=f"{mm_val:+.1f} mm ({val_curr:+.3f} m)", font=("Courier", 8), fg="#10B981", bg="#0B1120")
                    lbl_val.pack(anchor="w")
                    self.joint_readouts.append(lbl_val)

                    slider = tk.Scale(
                        f, from_=mm_min, to=mm_max, resolution=1.0, orient=tk.HORIZONTAL,
                        bg="#0B1120", fg="#94A3B8", activebackground="#10B981",
                        troughcolor="#111B33", highlightthickness=0, showvalue=False,
                        command=lambda v, idx=i: self._on_joint_slider_prismatic(idx, float(v))
                    )
                    slider.set(mm_val)
                    slider.pack(fill=tk.X, pady=2)
                    self.joint_sliders.append(slider)
        else:
            # Parallel Stewart Hexapod controls
            tk.Label(self.joint_scroll_frame, text="Translation Platform Coordinates:", font=("Helvetica", 9, "bold"), fg="#38BDF8", bg="#070A13").pack(anchor="w", pady=(4, 2))

            for axis, val, min_v, max_v in [("X", self.plat_trans.x, -0.15, 0.15), ("Y", self.plat_trans.y, -0.15, 0.15), ("Z", self.plat_trans.z, -0.10, 0.10)]:
                f = tk.Frame(self.joint_scroll_frame, bg="#0B1120", padx=8, pady=4, highlightbackground="#1E293B", highlightthickness=1)
                f.pack(fill=tk.X, pady=2, padx=4)
                lbl = tk.Label(f, text=f"Translate {axis}: {val*1000:+.1f} mm", font=("Courier", 8), fg="#F1F5F9", bg="#0B1120")
                lbl.pack(anchor="w")
                slider = tk.Scale(
                    f, from_=min_v * 1000, to=max_v * 1000, resolution=1.0, orient=tk.HORIZONTAL,
                    bg="#0B1120", fg="#94A3B8", activebackground="#38BDF8",
                    troughcolor="#111B33", highlightthickness=0, showvalue=False,
                    command=lambda v, ax=axis, l=lbl: self._on_hexapod_trans(ax, float(v), l)
                )
                slider.set(val * 1000)
                slider.pack(fill=tk.X)

            tk.Label(self.joint_scroll_frame, text="Orientation Euler Angles (RPY):", font=("Helvetica", 9, "bold"), fg="#10B981", bg="#070A13").pack(anchor="w", pady=(10, 2))

            for idx, name, val in [(0, "Roll", self.plat_rpy[0]), (1, "Pitch", self.plat_rpy[1]), (2, "Yaw", self.plat_rpy[2])]:
                f = tk.Frame(self.joint_scroll_frame, bg="#0B1120", padx=8, pady=4, highlightbackground="#1E293B", highlightthickness=1)
                f.pack(fill=tk.X, pady=2, padx=4)
                lbl = tk.Label(f, text=f"{name}: {math.degrees(val):+.1f} deg", font=("Courier", 8), fg="#F1F5F9", bg="#0B1120")
                lbl.pack(anchor="w")
                slider = tk.Scale(
                    f, from_=-25.0, to=25.0, resolution=0.5, orient=tk.HORIZONTAL,
                    bg="#0B1120", fg="#94A3B8", activebackground="#10B981",
                    troughcolor="#111B33", highlightthickness=0, showvalue=False,
                    command=lambda v, i=idx, l=lbl, n=name: self._on_hexapod_rot(i, float(v), l, n)
                )
                slider.set(math.degrees(val))
                slider.pack(fill=tk.X)

    def _build_tab_ik(self):
        """Construct controls for target coordinate positioning and DLS solver tuning."""
        frame = tk.Frame(self.tab_ik, bg="#070A13", padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="3D Spatial Target Coordinates", font=("Helvetica", 10, "bold"), fg="#38BDF8", bg="#070A13").pack(anchor="w", pady=(0, 6))

        # Target X, Y, Z Sliders
        self.target_sliders: Dict[str, tk.Scale] = {}
        self.target_labels: Dict[str, tk.Label] = {}

        for axis, default_val in [("X", self.target_pos.x), ("Y", self.target_pos.y), ("Z", self.target_pos.z)]:
            sub = tk.Frame(frame, bg="#0B1120", padx=8, pady=6, highlightbackground="#1E293B", highlightthickness=1)
            sub.pack(fill=tk.X, pady=4)

            lbl = tk.Label(sub, text=f"Target {axis}: {default_val:+.3f} m ({default_val*1000:+.1f} mm)", font=("Courier", 9, "bold"), fg="#F43F5E", bg="#0B1120")
            lbl.pack(anchor="w")
            self.target_labels[axis] = lbl

            slider = tk.Scale(
                sub, from_=-0.80, to=0.80, resolution=0.005, orient=tk.HORIZONTAL,
                bg="#0B1120", fg="#94A3B8", activebackground="#F43F5E",
                troughcolor="#111B33", highlightthickness=0, showvalue=False,
                command=lambda v, ax=axis: self._on_target_slider(ax, float(v))
            )
            slider.set(default_val)
            slider.pack(fill=tk.X, pady=2)
            self.target_sliders[axis] = slider

        # DLS Algorithm Parameters
        tk.Label(frame, text="Damped Least-Squares (DLS) Parameters", font=("Helvetica", 10, "bold"), fg="#F59E0B", bg="#070A13").pack(anchor="w", pady=(14, 6))

        damp_frame = tk.Frame(frame, bg="#0B1120", padx=8, pady=6, highlightbackground="#1E293B", highlightthickness=1)
        damp_frame.pack(fill=tk.X, pady=4)

        self.lbl_damping = tk.Label(damp_frame, text=f"Damping lambda: {self.ik_damping:.3f}", font=("Courier", 8), fg="#F59E0B", bg="#0B1120")
        self.lbl_damping.pack(anchor="w")
        slider_damp = tk.Scale(
            damp_frame, from_=0.01, to=0.30, resolution=0.005, orient=tk.HORIZONTAL,
            bg="#0B1120", fg="#94A3B8", activebackground="#F59E0B",
            troughcolor="#111B33", highlightthickness=0, showvalue=False,
            command=self._on_damping_change
        )
        slider_damp.set(self.ik_damping)
        slider_damp.pack(fill=tk.X, pady=2)

        # Options toggles
        opt_frame = tk.Frame(frame, bg="#070A13", pady=6)
        opt_frame.pack(fill=tk.X)

        self.chk_auto_ik_var = tk.BooleanVar(value=self.auto_solve_ik)
        chk_auto = tk.Checkbutton(
            opt_frame, text="Auto-Solve IK on Target Drag", variable=self.chk_auto_ik_var,
            bg="#070A13", fg="#38BDF8", selectcolor="#0B1120", activebackground="#070A13",
            font=("Helvetica", 9), command=self._toggle_auto_ik
        )
        chk_auto.pack(anchor="w")

        self.chk_ellip_var = tk.BooleanVar(value=self.show_ellipsoid)
        chk_ellip = tk.Checkbutton(
            opt_frame, text="Show Manipulability Ellipsoid", variable=self.chk_ellip_var,
            bg="#070A13", fg="#F59E0B", selectcolor="#0B1120", activebackground="#070A13",
            font=("Helvetica", 9), command=self._toggle_ellipsoid
        )
        chk_ellip.pack(anchor="w")

        # Solve IK Trigger Button
        btn_solve = tk.Button(
            frame, text="Solve DLS Inverse Kinematics", command=self._solve_ik_step,
            bg="#1E293B", fg="#38BDF8", activebackground="#38BDF8", activeforeground="#0B1120",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT, pady=6
        )
        btn_solve.pack(fill=tk.X, pady=(10, 4))

    def _build_tab_traj(self):
        """Construct automated continuous path generation tools."""
        frame = tk.Frame(self.tab_traj, bg="#070A13", padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Continuous Path Tracking", font=("Helvetica", 10, "bold"), fg="#38BDF8", bg="#070A13").pack(anchor="w", pady=(0, 6))

        # Path shape selector
        tk.Label(frame, text="Trajectory Pattern:", font=("Helvetica", 9), fg="#94A3B8", bg="#070A13").pack(anchor="w")
        self.traj_combo = ttk.Combobox(
            frame,
            values=["circle: Horizontal Circle in XY", "figure8: Lissajous Figure-Eight", "square: Orthogonal Square Box", "spiral: Ascending Helical Spiral"],
            state="readonly"
        )
        self.traj_combo.current(0)
        self.traj_combo.bind("<<ComboboxSelected>>", self._on_traj_shape_change)
        self.traj_combo.pack(fill=tk.X, pady=(2, 10))

        # Speed control
        tk.Label(frame, text="Tracking Frequency & Speed:", font=("Helvetica", 9), fg="#94A3B8", bg="#070A13").pack(anchor="w")
        slider_speed = tk.Scale(
            frame, from_=0.2, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg="#0B1120", fg="#94A3B8", activebackground="#38BDF8",
            troughcolor="#111B33", highlightthickness=0, showvalue=True,
            command=self._on_traj_speed_change
        )
        slider_speed.set(1.0)
        slider_speed.pack(fill=tk.X, pady=(2, 10))

        # Play / Pause button
        self.btn_traj_toggle = tk.Button(
            frame, text="Start Trajectory Tracking", command=self._toggle_trajectory,
            bg="#111B33", fg="#10B981", activebackground="#10B981", activeforeground="#070A13",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT, pady=6
        )
        self.btn_traj_toggle.pack(fill=tk.X, pady=(4, 6))

        btn_clear_trail = tk.Button(
            frame, text="Clear Motion Trail", command=self._clear_trail,
            bg="#111B33", fg="#F1F5F9", activebackground="#1E293B",
            font=("Helvetica", 9), relief=tk.FLAT, pady=4
        )
        btn_clear_trail.pack(fill=tk.X, pady=4)

    def _bind_events(self):
        """Bind mouse and keyboard interactions for the 3D viewport."""
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonPress-2>", self._on_mouse_press)
        self.canvas.bind("<B2-Motion>", self._on_mouse_pan)
        self.canvas.bind("<ButtonPress-3>", self._on_mouse_press)
        self.canvas.bind("<B3-Motion>", self._on_mouse_pan)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # macOS / Windows
        self.canvas.bind("<Button-4>", lambda e: self._zoom(0.90))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.10))

    def _on_mouse_press(self, event):
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

    def _on_mouse_drag(self, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        if self.interaction_mode == "orbit":
            self.camera.azimuth += dx * 0.008
            self.camera.elevation = max(-math.pi / 2 + 0.05, min(math.pi / 2 - 0.05, self.camera.elevation + dy * 0.008))
        elif self.interaction_mode == "target_drag" and self.is_serial:
            # Move 3D target coordinates relative to camera orientation
            eye = self.camera.get_eye_position()
            forward = (self.camera.center - eye).normalized()
            up_world = Vector3(0.0, 0.0, 1.0)
            right = forward.cross(up_world).normalized()
            up = right.cross(forward).normalized()

            scale = 0.0015 * self.camera.distance
            delta_target = (right * (dx * scale)) + (up * (-dy * scale))
            self.target_pos = self.target_pos + delta_target

            # Sync sliders
            if "X" in self.target_sliders:
                self.target_sliders["X"].set(self.target_pos.x)
                self.target_sliders["Y"].set(self.target_pos.y)
                self.target_sliders["Z"].set(self.target_pos.z)
                self._update_target_labels()

            if self.auto_solve_ik:
                self._solve_ik_step()

    def _on_mouse_pan(self, event):
        dx = event.x - self.mouse_last_x
        dy = event.y - self.mouse_last_y
        self.mouse_last_x = event.x
        self.mouse_last_y = event.y

        scale = 0.0012 * self.camera.distance
        eye = self.camera.get_eye_position()
        forward = (self.camera.center - eye).normalized()
        right = forward.cross(Vector3(0.0, 0.0, 1.0)).normalized()
        up = right.cross(forward).normalized()

        self.camera.center = self.camera.center - (right * (dx * scale)) + (up * (dy * scale))

    def _on_mouse_wheel(self, event):
        factor = 0.90 if event.delta > 0 else 1.10
        self._zoom(factor)

    def _zoom(self, factor: float):
        self.camera.distance = max(0.5, min(6.0, self.camera.distance * factor))

    def _reset_camera(self):
        self.camera.azimuth = 0.85
        self.camera.elevation = 0.45
        self.camera.distance = 2.4
        self.camera.center = Vector3(0.0, 0.0, 0.25)

    def _home_pose(self):
        if self.is_serial:
            self.joint_angles = list(self.metadata.get("initial_q", [0.0] * self.system.num_joints))
            self._sync_joint_sliders()
            pos, _ = self.system.get_end_effector_pose(self.joint_angles)
            self.target_pos = pos
            self._sync_target_sliders()
        else:
            self.plat_trans = Vector3(0.0, 0.0, 0.0)
            self.plat_rpy = [0.0, 0.0, 0.0]
            self._build_tab_joints()
        self.trail_buffer.clear()

    def _update_mouse_mode(self):
        self.interaction_mode = self.mouse_mode_var.get()

    def _on_preset_change(self, event):
        sel = self.preset_combo.get().split(":")[0].strip()
        self._load_preset(sel)
        self.lbl_model_title.configure(text=self.metadata.get("name", "Robot Manipulator"))
        self.lbl_model_desc.configure(text=self.metadata.get("description", ""))
        self._build_tab_joints()
        self._reset_camera()

    def _on_joint_slider(self, index: int, deg_val: float):
        rad_val = math.radians(deg_val)
        if index < len(self.joint_angles):
            self.joint_angles[index] = rad_val
            if index < len(self.joint_readouts):
                self.joint_readouts[index].configure(text=f"{deg_val:+.1f} deg ({rad_val:+.3f} rad)")

    def _on_joint_slider_prismatic(self, index: int, mm_val: float):
        m_val = mm_val / 1000.0
        if index < len(self.joint_angles):
            self.joint_angles[index] = m_val
            if index < len(self.joint_readouts):
                self.joint_readouts[index].configure(text=f"{mm_val:+.1f} mm ({m_val:+.3f} m)")

    def _on_target_slider(self, axis: str, val: float):
        if axis == "X":
            self.target_pos = Vector3(val, self.target_pos.y, self.target_pos.z)
        elif axis == "Y":
            self.target_pos = Vector3(self.target_pos.x, val, self.target_pos.z)
        elif axis == "Z":
            self.target_pos = Vector3(self.target_pos.x, self.target_pos.y, val)
        self._update_target_labels()
        if self.auto_solve_ik and self.is_serial:
            self._solve_ik_step()

    def _update_target_labels(self):
        if "X" in self.target_labels:
            self.target_labels["X"].configure(text=f"Target X: {self.target_pos.x:+.3f} m ({self.target_pos.x*1000:+.1f} mm)")
            self.target_labels["Y"].configure(text=f"Target Y: {self.target_pos.y:+.3f} m ({self.target_pos.y*1000:+.1f} mm)")
            self.target_labels["Z"].configure(text=f"Target Z: {self.target_pos.z:+.3f} m ({self.target_pos.z*1000:+.1f} mm)")

    def _on_hexapod_trans(self, axis: str, mm_val: float, label: tk.Label):
        m_val = mm_val / 1000.0
        label.configure(text=f"Translate {axis}: {mm_val:+.1f} mm")
        if axis == "X":
            self.plat_trans = Vector3(m_val, self.plat_trans.y, self.plat_trans.z)
        elif axis == "Y":
            self.plat_trans = Vector3(self.plat_trans.x, m_val, self.plat_trans.z)
        elif axis == "Z":
            self.plat_trans = Vector3(self.plat_trans.x, self.plat_trans.y, m_val)

    def _on_hexapod_rot(self, idx: int, deg_val: float, label: tk.Label, name: str):
        label.configure(text=f"{name}: {deg_val:+.1f} deg")
        self.plat_rpy[idx] = math.radians(deg_val)

    def _on_damping_change(self, val: float):
        self.ik_damping = float(val)
        self.lbl_damping.configure(text=f"Damping lambda: {self.ik_damping:.3f}")

    def _toggle_auto_ik(self):
        self.auto_solve_ik = self.chk_auto_ik_var.get()

    def _toggle_ellipsoid(self):
        self.show_ellipsoid = self.chk_ellip_var.get()

    def _on_traj_shape_change(self, event):
        self.traj_type = self.traj_combo.get().split(":")[0].strip()

    def _on_traj_speed_change(self, val: float):
        self.traj_speed = float(val)

    def _toggle_trajectory(self):
        self.traj_active = not self.traj_active
        if self.traj_active:
            self.btn_traj_toggle.configure(text="Pause Trajectory Tracking", fg="#F43F5E")
        else:
            self.btn_traj_toggle.configure(text="Start Trajectory Tracking", fg="#10B981")

    def _clear_trail(self):
        self.trail_buffer.clear()

    def _solve_ik_step(self):
        """Execute Damped Least-Squares inverse kinematics iteration."""
        if not self.is_serial:
            return

        solved_q, converged, err = self.system.solve_inverse_kinematics_dls(
            self.target_pos,
            initial_angles=self.joint_angles,
            max_iter=self.ik_max_iter,
            pos_tolerance=self.ik_tolerance,
            damping=self.ik_damping,
        )
        self.joint_angles = solved_q
        self._sync_joint_sliders()

    def _sync_joint_sliders(self):
        """Update joint slider positions from joint_angles array."""
        for i, slider in enumerate(self.joint_sliders):
            if i < len(self.joint_angles) and i < len(self.system.links):
                link = self.system.links[i]
                if link.is_revolute:
                    deg = math.degrees(self.joint_angles[i])
                    slider.set(deg)
                    if i < len(self.joint_readouts):
                        self.joint_readouts[i].configure(text=f"{deg:+.1f} deg ({self.joint_angles[i]:+.3f} rad)")
                else:
                    mm = self.joint_angles[i] * 1000.0
                    slider.set(mm)
                    if i < len(self.joint_readouts):
                        self.joint_readouts[i].configure(text=f"{mm:+.1f} mm ({self.joint_angles[i]:+.3f} m)")

    def _sync_target_sliders(self):
        if "X" in self.target_sliders:
            self.target_sliders["X"].set(self.target_pos.x)
            self.target_sliders["Y"].set(self.target_pos.y)
            self.target_sliders["Z"].set(self.target_pos.z)
            self._update_target_labels()

    def _update_trajectory(self, dt: float):
        """Update analytical path coordinate when trajectory generator is running."""
        if not self.traj_active:
            return

        self.traj_time += dt * self.traj_speed
        t = self.traj_time

        center_x, center_y, center_z = 0.35, 0.15, 0.32
        radius = 0.15

        if self.traj_type == "circle":
            x = center_x + radius * math.cos(t)
            y = center_y + radius * math.sin(t)
            z = center_z
        elif self.traj_type == "figure8":
            x = center_x + radius * math.sin(t)
            y = center_y + radius * math.sin(2.0 * t) * 0.6
            z = center_z + 0.05 * math.cos(t)
        elif self.traj_type == "square":
            cycle = t % 4.0
            half = radius * 0.8
            if cycle < 1.0:
                s = cycle
                x = center_x - half + 2.0 * half * s
                y = center_y - half
            elif cycle < 2.0:
                s = cycle - 1.0
                x = center_x + half
                y = center_y - half + 2.0 * half * s
            elif cycle < 3.0:
                s = cycle - 2.0
                x = center_x + half - 2.0 * half * s
                y = center_y + half
            else:
                s = cycle - 3.0
                x = center_x - half
                y = center_y + half - 2.0 * half * s
            z = center_z
        elif self.traj_type == "spiral":
            x = center_x + radius * math.cos(t) * (0.6 + 0.4 * math.sin(t * 0.3))
            y = center_y + radius * math.sin(t) * (0.6 + 0.4 * math.sin(t * 0.3))
            z = center_z + 0.08 * math.sin(t * 0.5)
        else:
            x, y, z = center_x, center_y, center_z

        self.target_pos = Vector3(x, y, z)
        self._sync_target_sliders()
        if self.is_serial:
            self._solve_ik_step()

    def _render_scene(self):
        """Draw complete 3D scene onto the Tkinter canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # 1. Draw 3D Isometric Ground Grid
        self._render_ground_grid(w, h)

        # 2. Draw 3D Base Axes Triad
        self._render_coordinate_triad(Vector3(0.0, 0.0, 0.0), Matrix4(), w, h, axis_len=0.15, prefix="w")

        if self.is_serial:
            # 3. Compute Serial Forward Kinematics
            transforms = self.system.forward_kinematics(self.joint_angles)
            joint_positions = [t.translation for t in transforms]
            p_end = joint_positions[-1]

            # Record motion trail
            if self.show_trail:
                self.trail_buffer.append(p_end)
                if len(self.trail_buffer) > self.max_trail_points:
                    self.trail_buffer.pop(0)
                self._render_trail(w, h)

            # 4. Render Target Marker & Ground Projection
            self._render_target_marker(w, h)

            # 5. Render Manipulability Ellipsoid
            if self.show_ellipsoid:
                self._render_manipulability_ellipsoid(p_end, w, h)

            # 6. Render Serial Robot Arm Links and Joint Nodes
            self._render_robot_links(joint_positions, transforms, w, h)

            # 7. Render End-Effector Tool Frame
            self._render_coordinate_triad(p_end, transforms[-1], w, h, axis_len=0.08, prefix="e")

            # 8. Update Telemetry Bar
            err = (self.target_pos - p_end).norm() * 1000.0
            w_dex = self.system.compute_manipulability(self.joint_angles)
            status_text = "Tracking" if self.traj_active else ("Converged" if err < 3.0 else "Solving")
            self.lbl_telemetry.configure(
                text=f"End-Effector: X:{p_end.x:+.3f}m Y:{p_end.y:+.3f}m Z:{p_end.z:+.3f}m | Error: {err:4.1f}mm | Dexterity w: {w_dex:.4f} | State: {status_text}"
            )
        else:
            # Parallel Stewart Hexapod Rendering
            lengths, plat_pts = self.system.inverse_kinematics(self.plat_trans, self.plat_rpy[0], self.plat_rpy[1], self.plat_rpy[2])
            self._render_stewart_hexapod(lengths, plat_pts, w, h)

            l_min = min(lengths) * 1000.0
            l_max = max(lengths) * 1000.0
            self.lbl_telemetry.configure(
                text=f"Stewart Platform | Struts L1..L6: [{lengths[0]*1000:.0f}, {lengths[1]*1000:.0f}, {lengths[2]*1000:.0f}, {lengths[3]*1000:.0f}, {lengths[4]*1000:.0f}, {lengths[5]*1000:.0f}] mm | Range: {l_min:.0f} to {l_max:.0f} mm"
            )

    def _render_ground_grid(self, w: float, h: float):
        """Draw concentric rings and ground plane grid."""
        grid_half = 0.8
        step = 0.2

        # Grid lines in X and Y directions
        curr = -grid_half
        while curr <= grid_half + 1e-4:
            # Line along X (varying Y)
            p1 = self.camera.project_point(Vector3(curr, -grid_half, 0.0), w, h)
            p2 = self.camera.project_point(Vector3(curr, grid_half, 0.0), w, h)
            if p1 and p2:
                color = "#1E293B" if abs(curr) > 1e-4 else "#334155"
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=1)

            # Line along Y (varying X)
            q1 = self.camera.project_point(Vector3(-grid_half, curr, 0.0), w, h)
            q2 = self.camera.project_point(Vector3(grid_half, curr, 0.0), w, h)
            if q1 and q2:
                color = "#1E293B" if abs(curr) > 1e-4 else "#334155"
                self.canvas.create_line(q1[0], q1[1], q2[0], q2[1], fill=color, width=1)

            curr += step

        # Outer boundary workspace circle
        circle_segs = 36
        pts: List[Tuple[float, float]] = []
        for s in range(circle_segs + 1):
            ang = s * (2.0 * math.pi / circle_segs)
            pt = self.camera.project_point(Vector3(grid_half * math.cos(ang), grid_half * math.sin(ang), 0.0), w, h)
            if pt:
                pts.append((pt[0], pt[1]))

        for i in range(len(pts) - 1):
            self.canvas.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="#1E293B", dash=(3, 3), width=1)

    def _render_coordinate_triad(
        self, origin: Vector3, transform: Matrix4, w: float, h: float, axis_len: float = 0.12, prefix: str = ""
    ):
        """Draw RGB Cartesian coordinate frame axes."""
        o_proj = self.camera.project_point(origin, w, h)
        if not o_proj:
            return

        x_dir = transform.x_axis * axis_len
        y_dir = transform.y_axis * axis_len
        z_dir = transform.z_axis * axis_len

        px = self.camera.project_point(origin + x_dir, w, h)
        py = self.camera.project_point(origin + y_dir, w, h)
        pz = self.camera.project_point(origin + z_dir, w, h)

        if px:
            self.canvas.create_line(o_proj[0], o_proj[1], px[0], px[1], fill="#EF4444", width=2, arrow=tk.LAST)
            self.canvas.create_text(px[0] + 5, px[1], text=f"{prefix}X", fill="#EF4444", font=("Helvetica", 7, "bold"))
        if py:
            self.canvas.create_line(o_proj[0], o_proj[1], py[0], py[1], fill="#10B981", width=2, arrow=tk.LAST)
            self.canvas.create_text(py[0] + 5, py[1], text=f"{prefix}Y", fill="#10B981", font=("Helvetica", 7, "bold"))
        if pz:
            self.canvas.create_line(o_proj[0], o_proj[1], pz[0], pz[1], fill="#38BDF8", width=2, arrow=tk.LAST)
            self.canvas.create_text(pz[0], pz[1] - 6, text=f"{prefix}Z", fill="#38BDF8", font=("Helvetica", 7, "bold"))

    def _render_target_marker(self, w: float, h: float):
        """Draw 3D target coordinate sphere, ground projection line, and reticle."""
        t_proj = self.camera.project_point(self.target_pos, w, h)
        ground_proj = self.camera.project_point(Vector3(self.target_pos.x, self.target_pos.y, 0.0), w, h)

        if ground_proj and t_proj:
            # Vertical drop line to ground
            self.canvas.create_line(ground_proj[0], ground_proj[1], t_proj[0], t_proj[1], fill="#F43F5E", dash=(2, 2), width=1)
            # Ground footprint ring
            r_g = 5.0
            self.canvas.create_oval(ground_proj[0] - r_g, ground_proj[1] - r_g, ground_proj[0] + r_g, ground_proj[1] + r_g, outline="#F43F5E", width=1)

        if t_proj:
            sx, sy, z_depth = t_proj
            r = max(4.0, 14.0 / max(0.5, z_depth))

            # Outer glowing halo
            self.canvas.create_oval(sx - r * 1.5, sy - r * 1.5, sx + r * 1.5, sy + r * 1.5, outline="#F43F5E", width=1, dash=(3, 2))
            # Core target sphere
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#F43F5E", outline="#FFFFFF", width=1)
            # Crosshair reticle
            self.canvas.create_line(sx - r * 1.8, sy, sx + r * 1.8, sy, fill="#F43F5E", width=1)
            self.canvas.create_line(sx, sy - r * 1.8, sx, sy + r * 1.8, fill="#F43F5E", width=1)
            self.canvas.create_text(sx + r + 4, sy - 8, text="Target", fill="#F43F5E", font=("Helvetica", 8, "bold"), anchor="w")

    def _render_trail(self, w: float, h: float):
        """Draw breadcrumb motion trail behind end-effector."""
        if len(self.trail_buffer) < 2:
            return

        screen_pts: List[Tuple[float, float]] = []
        for pt in self.trail_buffer:
            p = self.camera.project_point(pt, w, h)
            if p:
                screen_pts.append((p[0], p[1]))

        n = len(screen_pts)
        for i in range(n - 1):
            alpha = (i + 1) / n
            color = "#00F0FF" if alpha > 0.7 else ("#0284C7" if alpha > 0.3 else "#075985")
            self.canvas.create_line(
                screen_pts[i][0], screen_pts[i][1], screen_pts[i+1][0], screen_pts[i+1][1],
                fill=color, width=max(1, int(alpha * 2.5))
            )

    def _render_manipulability_ellipsoid(self, center: Vector3, w: float, h: float):
        """Draw 3D velocity dexterity ellipsoid at end-effector pose."""
        j = self.system.compute_jacobian(self.joint_angles)
        # Compute 3x3 position Gram matrix G = J_v * J_v^T
        g = [[0.0] * 3 for _ in range(3)]
        n = self.system.num_joints
        for r1 in range(3):
            for r2 in range(3):
                g[r1][r2] = sum(j[r1][k] * j[r2][k] for k in range(n))

        # Scale factor for visual representation
        scale = 0.18
        rx = math.sqrt(max(0.001, g[0][0])) * scale
        ry = math.sqrt(max(0.001, g[1][1])) * scale
        rz = math.sqrt(max(0.001, g[2][2])) * scale

        # Wireframe latitude/longitude rings
        num_pts = 20
        # XY ring
        xy_pts: List[Tuple[float, float]] = []
        for i in range(num_pts + 1):
            th = i * 2.0 * math.pi / num_pts
            pt = center + Vector3(rx * math.cos(th), ry * math.sin(th), 0.0)
            p_proj = self.camera.project_point(pt, w, h)
            if p_proj:
                xy_pts.append((p_proj[0], p_proj[1]))
        for i in range(len(xy_pts) - 1):
            self.canvas.create_line(xy_pts[i][0], xy_pts[i][1], xy_pts[i+1][0], xy_pts[i+1][1], fill="#F59E0B", width=1)

        # XZ ring
        xz_pts: List[Tuple[float, float]] = []
        for i in range(num_pts + 1):
            th = i * 2.0 * math.pi / num_pts
            pt = center + Vector3(rx * math.cos(th), 0.0, rz * math.sin(th))
            p_proj = self.camera.project_point(pt, w, h)
            if p_proj:
                xz_pts.append((p_proj[0], p_proj[1]))
        for i in range(len(xz_pts) - 1):
            self.canvas.create_line(xz_pts[i][0], xz_pts[i][1], xz_pts[i+1][0], xz_pts[i+1][1], fill="#F59E0B", width=1, dash=(2, 2))

    def _render_robot_links(
        self, positions: List[Vector3], transforms: List[Matrix4], w: float, h: float
    ):
        """Render robot link segments with depth-sorted cylinders and joint spheres."""
        num_joints = len(positions)

        # Draw base mounting pedestal
        p0 = positions[0]
        base_top = self.camera.project_point(p0, w, h)
        base_bot = self.camera.project_point(Vector3(p0.x, p0.y, 0.0), w, h)
        if base_top and base_bot:
            self.canvas.create_line(base_bot[0], base_bot[1], base_top[0], base_top[1], fill="#475569", width=10)
            self.canvas.create_oval(base_bot[0] - 12, base_bot[1] - 4, base_bot[0] + 12, base_bot[1] + 4, fill="#334155", outline="#64748B")

        # Draw serial links
        for i in range(num_joints - 1):
            pt_a = positions[i]
            pt_b = positions[i + 1]

            p_a = self.camera.project_point(pt_a, w, h)
            p_b = self.camera.project_point(pt_b, w, h)

            if p_a and p_b:
                # Link segment thickness and color based on index
                link_width = max(3, int(10 - i * 1.2))
                color = "#38BDF8" if i % 2 == 0 else "#60A5FA"
                self.canvas.create_line(p_a[0], p_a[1], p_b[0], p_b[1], fill="#1E293B", width=link_width + 4)
                self.canvas.create_line(p_a[0], p_a[1], p_b[0], p_b[1], fill=color, width=link_width)

        # Draw joint nodes
        for i, pos in enumerate(positions):
            p = self.camera.project_point(pos, w, h)
            if p:
                sx, sy, z_depth = p
                r = max(3.0, 10.0 / max(0.4, z_depth))
                j_color = "#E2E8F0" if i == 0 else ("#10B981" if i == num_joints - 1 else "#38BDF8")
                self.canvas.create_oval(sx - r - 1, sy - r - 1, sx + r + 1, sy + r + 1, fill="#0F172A", outline="")
                self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=j_color, outline="#FFFFFF", width=1)
                self.canvas.create_text(sx, sy - r - 6, text=f"J{i}", fill="#94A3B8", font=("Helvetica", 7))

    def _render_stewart_hexapod(
        self, leg_lengths: List[float], plat_pts: List[Vector3], w: float, h: float
    ):
        """Render 6-DOF parallel Stewart platform with base, platform, and 6 actuator legs."""
        base_pts = self.system.base_anchors

        # Project base anchor points
        proj_base: List[Tuple[float, float]] = []
        for bp in base_pts:
            p = self.camera.project_point(bp, w, h)
            if p:
                proj_base.append((p[0], p[1]))

        # Project platform joints
        proj_plat: List[Tuple[float, float]] = []
        for pp in plat_pts:
            p = self.camera.project_point(pp, w, h)
            if p:
                proj_plat.append((p[0], p[1]))

        # Draw base plate polygon
        if len(proj_base) == 6:
            base_coords = [coord for pt in proj_base for coord in pt]
            self.canvas.create_polygon(base_coords, fill="#0F172A", outline="#475569", width=2)

        # Draw mobile top platform polygon
        if len(proj_plat) == 6:
            plat_coords = [coord for pt in proj_plat for coord in pt]
            self.canvas.create_polygon(plat_coords, fill="#1E293B", outline="#38BDF8", width=2)
            # Center of platform
            center_world = self.plat_trans + Vector3(0.0, 0.0, self.system.home_height)
            cp = self.camera.project_point(center_world, w, h)
            if cp:
                self.canvas.create_oval(cp[0] - 4, cp[1] - 4, cp[0] + 4, cp[1] + 4, fill="#F43F5E", outline="#FFFFFF")
                self.canvas.create_text(cp[0], cp[1] - 8, text="Payload Stage", fill="#38BDF8", font=("Helvetica", 8, "bold"))

        # Draw 6 actuator struts
        for i in range(6):
            if i < len(proj_base) and i < len(proj_plat):
                p_base = proj_base[i]
                p_plat = proj_plat[i]

                # Strain color: nominal stroke ~0.45m
                length = leg_lengths[i]
                strain = (length - 0.40) / 0.15
                if strain < 0.2:
                    strut_color = "#10B981"
                elif strain < 0.6:
                    strut_color = "#F59E0B"
                else:
                    strut_color = "#EF4444"

                # Outer cylinder and inner piston
                self.canvas.create_line(p_base[0], p_base[1], p_plat[0], p_plat[1], fill="#0F172A", width=8)
                self.canvas.create_line(p_base[0], p_base[1], p_plat[0], p_plat[1], fill=strut_color, width=4)

                # Strut number label
                mid_x = 0.5 * (p_base[0] + p_plat[0])
                mid_y = 0.5 * (p_base[1] + p_plat[1])
                self.canvas.create_text(mid_x + 6, mid_y, text=f"L{i+1}:{length*1000:.0f}", fill="#94A3B8", font=("Courier", 7))

    def _animate(self):
        """Master render loop updating trajectory and canvas at 60 FPS."""
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now

        if dt > 0.001:
            instant_fps = 1.0 / dt
            self.fps = 0.92 * self.fps + 0.08 * instant_fps
            self.lbl_fps.configure(text=f"FPS: {self.fps:4.1f}")

        # Update trajectory path
        if self.traj_active:
            self._update_trajectory(min(0.05, dt))

        # Render complete scene
        self._render_scene()

        # Schedule next frame in 16ms
        self.root.after(16, self._animate)


def main():
    root = tk.Tk()
    app = KineMatixApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
