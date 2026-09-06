"""
Solida: High-Level Solid Modeling CAD Kernel & Assembly Orchestrator.
Provides fluent Part and Assembly APIs for parametric 3D CAD modeling,
transformations, CSG Booleans, interference detection, and multi-format export.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import os
import math
from typing import List, Dict, Tuple, Optional, Sequence, Union
from solida.geometry import Vector3D, Matrix4x4, Quaternion, BoundingBox3D, Plane
from solida.brep import Solid, build_solid_from_polygons
from solida.primitives import make_box, make_cylinder, make_sphere, make_cone, make_torus
from solida.features import Sketch2D, extrude, revolve, loft
from solida.csg import csg_union, csg_difference, csg_intersection
from solida.tessellation import (
    tessellate_solid,
    export_stl_ascii,
    export_stl_binary,
    export_obj,
    Facet3D,
)
from solida.visualizer import render_solid_cad, render_cad_hud


class Part:
    """
    Parametric Solid Model Part with fluent chaining API.
    Wraps an exact B-Rep manifold Solid and exposes boolean operators,
    affine transformations, mass properties, and multi-format exporters.
    """
    __slots__ = ("_solid", "density_g_cm3", "color_rgb")

    def __init__(
        self,
        solid: Solid,
        density_g_cm3: float = 2.7,  # Aluminum 6061 default: 2.70 g/cm^3
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> None:
        self._solid = solid
        self.density_g_cm3 = float(density_g_cm3)
        self.color_rgb = color_rgb

    @property
    def solid(self) -> Solid:
        return self._solid

    @property
    def name(self) -> str:
        return self._solid.name

    @name.setter
    def name(self, val: str) -> None:
        self._solid.name = str(val)

    # Factory constructors
    @classmethod
    def box(
        cls,
        dx: float,
        dy: float,
        dz: float,
        center: bool = False,
        origin: Optional[Vector3D] = None,
        name: str = "Box",
        density_g_cm3: float = 2.7,
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> Part:
        return cls(make_box(dx, dy, dz, center=center, origin=origin, name=name), density_g_cm3=density_g_cm3, color_rgb=color_rgb)

    @classmethod
    def cylinder(
        cls,
        radius: float,
        height: float,
        segments: int = 32,
        center: bool = False,
        name: str = "Cylinder",
        density_g_cm3: float = 2.7,
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> Part:
        return cls(make_cylinder(radius, height, segments=segments, center=center, name=name), density_g_cm3=density_g_cm3, color_rgb=color_rgb)

    @classmethod
    def sphere(
        cls,
        radius: float,
        rings: int = 16,
        sectors: int = 32,
        center: Optional[Vector3D] = None,
        name: str = "Sphere",
        density_g_cm3: float = 2.7,
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> Part:
        return cls(make_sphere(radius, rings=rings, sectors=sectors, center=center, name=name), density_g_cm3=density_g_cm3, color_rgb=color_rgb)

    @classmethod
    def cone(
        cls,
        radius: float,
        height: float,
        segments: int = 32,
        center: bool = False,
        name: str = "Cone",
        density_g_cm3: float = 2.7,
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> Part:
        return cls(make_cone(radius, height, segments=segments, center=center, name=name), density_g_cm3=density_g_cm3, color_rgb=color_rgb)

    @classmethod
    def torus(
        cls,
        major_radius: float,
        minor_radius: float,
        major_segs: int = 24,
        minor_segs: int = 16,
        name: str = "Torus",
        density_g_cm3: float = 2.7,
        color_rgb: Optional[Tuple[int, int, int]] = (0, 220, 255),
    ) -> Part:
        return cls(make_torus(major_radius, minor_radius, major_segs=major_segs, minor_segs=minor_segs, name=name), density_g_cm3=density_g_cm3, color_rgb=color_rgb)

    @classmethod
    def from_extrusion(
        cls,
        sketch: Sketch2D,
        direction: Optional[Vector3D] = None,
        height: Optional[float] = None,
        draft_angle_deg: float = 0.0,
        twist_angle_deg: float = 0.0,
        steps: int = 1,
        name: str = "Extrusion",
    ) -> Part:
        return cls(extrude(sketch, direction=direction, height=height, draft_angle_deg=draft_angle_deg, twist_angle_deg=twist_angle_deg, steps=steps, name=name))

    @classmethod
    def from_revolve(
        cls,
        sketch: Sketch2D,
        axis_origin: Optional[Vector3D] = None,
        axis_dir: Optional[Vector3D] = None,
        angle_deg: float = 360.0,
        segments: int = 32,
        name: str = "Revolve",
    ) -> Part:
        return cls(revolve(sketch, axis_origin=axis_origin, axis_dir=axis_dir, angle_deg=angle_deg, segments=segments, name=name))

    @classmethod
    def from_loft(
        cls,
        sketches: Sequence[Sketch2D],
        name: str = "Loft",
    ) -> Part:
        return cls(loft(sketches, name=name))

    # Geometric & Physical Properties
    @property
    def volume(self) -> float:
        """Volume in mm^3."""
        return self._solid.volume()

    @property
    def surface_area(self) -> float:
        """Surface area in mm^2."""
        return self._solid.surface_area()

    @property
    def mass_grams(self) -> float:
        """Mass in grams = Volume (cm^3) * Density (g/cm^3). 1 cm^3 = 1000 mm^3."""
        return (self.volume / 1000.0) * self.density_g_cm3

    @property
    def centroid(self) -> Vector3D:
        return self._solid.center_of_mass()

    @property
    def bounding_box(self) -> BoundingBox3D:
        return self._solid.bounding_box()

    @property
    def euler_characteristic(self) -> int:
        return self._solid.outer_shell.euler_characteristic()

    @property
    def is_manifold(self) -> bool:
        return all(e.is_manifold for e in self._solid.edges)

    # Spatial Transformations (returns new transformed Part)
    def transform(self, matrix: Matrix4x4) -> Part:
        """Applies arbitrary 4x4 homogeneous affine transformation matrix."""
        new_polys: List[List[Vector3D]] = []
        for face in self._solid.faces:
            poly = [matrix.transform_point(v.point) for v in face.vertices()]
            new_polys.append(poly)
        new_solid = build_solid_from_polygons(new_polys, name=self.name)
        return Part(new_solid, density_g_cm3=self.density_g_cm3, color_rgb=self.color_rgb)

    def translate(self, dx: float, dy: float, dz: float) -> Part:
        """Translates part by Cartesian displacement vector."""
        return self.transform(Matrix4x4.translation(dx, dy, dz))

    def rotate_x(self, angle_deg: float) -> Part:
        return self.transform(Matrix4x4.rotation_x(math.radians(angle_deg)))

    def rotate_y(self, angle_deg: float) -> Part:
        return self.transform(Matrix4x4.rotation_y(math.radians(angle_deg)))

    def rotate_z(self, angle_deg: float) -> Part:
        return self.transform(Matrix4x4.rotation_z(math.radians(angle_deg)))

    def rotate_axis(self, axis: Vector3D, angle_deg: float, origin: Optional[Vector3D] = None) -> Part:
        """Rotates part about arbitrary 3D axis and origin point."""
        theta = math.radians(angle_deg)
        if origin is None:
            mat = Matrix4x4.rotation_axis_angle(axis, theta)
        else:
            mat = (
                Matrix4x4.translation(origin.x, origin.y, origin.z)
                @ Matrix4x4.rotation_axis_angle(axis, theta)
                @ Matrix4x4.translation(-origin.x, -origin.y, -origin.z)
            )
        return self.transform(mat)

    def scale(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> Part:
        """Scales part along Cartesian axes."""
        sy_val = sx if sy is None else sy
        sz_val = sx if sz is None else sz
        return self.transform(Matrix4x4.scaling(sx, sy_val, sz_val))

    # Constructive Solid Geometry (CSG) Booleans
    def union(self, other: Part, name: Optional[str] = None) -> Part:
        """CSG Boolean Union (self + other)."""
        res_name = f"{self.name}_Union_{other.name}" if name is None else name
        res_solid = csg_union(self._solid, other._solid, name=res_name)
        return Part(res_solid, density_g_cm3=self.density_g_cm3, color_rgb=self.color_rgb)

    def difference(self, other: Part, name: Optional[str] = None) -> Part:
        """CSG Boolean Difference / Cut (self - other)."""
        res_name = f"{self.name}_Cut_{other.name}" if name is None else name
        res_solid = csg_difference(self._solid, other._solid, name=res_name)
        return Part(res_solid, density_g_cm3=self.density_g_cm3, color_rgb=self.color_rgb)

    def intersect(self, other: Part, name: Optional[str] = None) -> Part:
        """CSG Boolean Intersection (self & other)."""
        res_name = f"{self.name}_Intersect_{other.name}" if name is None else name
        res_solid = csg_intersection(self._solid, other._solid, name=res_name)
        return Part(res_solid, density_g_cm3=self.density_g_cm3, color_rgb=self.color_rgb)

    # Operator Overloading for intuitive mathematical scripting
    def __add__(self, other: Part) -> Part:
        return self.union(other)

    def __sub__(self, other: Part) -> Part:
        return self.difference(other)

    def __and__(self, other: Part) -> Part:
        return self.intersect(other)

    # Advanced CAD Operations
    def hollow(self, wall_thickness: float) -> Part:
        """
        Creates a hollowed thin-walled shell by subtracting a scaled/offset
        internal cavity from the solid.
        """
        t = abs(float(wall_thickness))
        bbox = self.bounding_box
        ext = bbox.extents()
        if min(ext.x, ext.y, ext.z) <= 2.0 * t:
            raise ValueError("Wall thickness exceeds solid bounding dimensions")

        # Scale factor per axis: (dim - 2t) / dim
        sx = (ext.x - 2.0 * t) / ext.x
        sy = (ext.y - 2.0 * t) / ext.y
        sz = (ext.z - 2.0 * t) / ext.z

        c = bbox.center()
        # Scale about bounding box center
        mat = (
            Matrix4x4.translation(c.x, c.y, c.z)
            @ Matrix4x4.scaling(sx, sy, sz)
            @ Matrix4x4.translation(-c.x, -c.y, -c.z)
        )
        inner = self.transform(mat)
        return self.difference(inner, name=f"{self.name}_Hollow")

    # Exporters
    def export_stl(self, file_path: str, binary: bool = True) -> int:
        """Exports part to STL file (binary or ASCII). Returns byte count."""
        if binary:
            data = export_stl_binary(self._solid, header_text=f"Solida CAD - {self.name}")
            with open(file_path, "wb") as f:
                f.write(data)
            return len(data)
        else:
            text = export_stl_ascii(self._solid, solid_name=self.name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            return len(text.encode("utf-8"))

    def export_obj(self, file_path: str) -> int:
        """Exports part to Wavefront OBJ format. Returns character count."""
        text = export_obj(self._solid, name=self.name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return len(text)

    # Sub-Pixel Terminal Visualization & Telemetry
    def show(
        self,
        azimuth_deg: float = 35.0,
        elevation_deg: float = 25.0,
        char_width: int = 68,
        char_height: int = 18,
        mode: str = "shaded",
        use_ansi_color: bool = True,
    ) -> str:
        """Returns rendered string containing telemetry HUD and sub-pixel Braille 3D graphic."""
        hud = render_cad_hud(self._solid, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        view = render_solid_cad(
            self._solid,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            char_width=char_width,
            char_height=char_height,
            mode=mode,
            use_ansi_color=use_ansi_color,
        )
        return f"{hud}\n{view}"


class Workplane:
    """
    Parametric 2D Workplane in 3D Space.
    Facilitates sketching profiles on arbitrary datum planes and sweeping them into 3D Parts.
    """
    __slots__ = ("origin", "normal", "x_dir", "y_dir")

    def __init__(
        self,
        origin: Optional[Vector3D] = None,
        normal: Optional[Vector3D] = None,
        x_dir: Optional[Vector3D] = None,
    ) -> None:
        self.origin = Vector3D(0, 0, 0) if origin is None else origin
        self.normal = Vector3D(0, 0, 1) if normal is None else normal.normalized()

        if x_dir is None:
            # Pick orthogonal x_dir
            cand = Vector3D(1, 0, 0) if abs(self.normal.x) < 0.9 else Vector3D(0, 1, 0)
            self.x_dir = self.normal.cross(cand).normalized()
        else:
            self.x_dir = x_dir.normalized()
        self.y_dir = self.normal.cross(self.x_dir).normalized()

    @classmethod
    def XY(cls, z_offset: float = 0.0) -> Workplane:
        return cls(origin=Vector3D(0, 0, z_offset), normal=Vector3D(0, 0, 1), x_dir=Vector3D(1, 0, 0))

    @classmethod
    def XZ(cls, y_offset: float = 0.0) -> Workplane:
        return cls(origin=Vector3D(0, y_offset, 0), normal=Vector3D(0, -1, 0), x_dir=Vector3D(1, 0, 0))

    @classmethod
    def YZ(cls, x_offset: float = 0.0) -> Workplane:
        return cls(origin=Vector3D(x_offset, 0, 0), normal=Vector3D(1, 0, 0), x_dir=Vector3D(0, 1, 0))

    def to_world(self, u: float, v: float) -> Vector3D:
        """Converts 2D (u, v) workplane coordinates into 3D world space."""
        return self.origin + self.x_dir * u + self.y_dir * v

    def map_sketch(self, sketch_2d: Sketch2D) -> Sketch2D:
        """Projects a 2D profile sketch in the local (u, v) plane into 3D world coordinates."""
        world_pts = [self.to_world(p.x, p.y) for p in sketch_2d.points]
        return Sketch2D(world_pts)

    def extrude_rect(
        self,
        width: float,
        height: float,
        depth: float,
        center: bool = False,
        name: str = "ExtrudedRect",
    ) -> Part:
        s2d = Sketch2D.rectangle(width, height, center=center)
        s3d = self.map_sketch(s2d)
        dir_vec = self.normal * depth
        return Part.from_extrusion(s3d, direction=dir_vec, name=name)

    def extrude_circle(
        self,
        radius: float,
        depth: float,
        segments: int = 32,
        name: str = "ExtrudedCircle",
    ) -> Part:
        s2d = Sketch2D.circle(radius, segments=segments)
        s3d = self.map_sketch(s2d)
        dir_vec = self.normal * depth
        return Part.from_extrusion(s3d, direction=dir_vec, name=name)


class Assembly:
    """
    Multi-Part Mechanical CAD Assembly.
    Manages spatial relationships, mass budget, bounding extents,
    part-to-part interference & clash detection, and composite OBJ export.
    """
    __slots__ = ("name", "_items")

    def __init__(self, name: str = "MainAssembly") -> None:
        self.name = name
        self._items: List[Tuple[str, Part, Matrix4x4]] = []

    def add_part(
        self,
        part: Part,
        instance_name: Optional[str] = None,
        transform: Optional[Matrix4x4] = None,
    ) -> Assembly:
        """Adds a part instance with an optional relative transformation matrix."""
        inst = f"{part.name}_{len(self._items) + 1}" if instance_name is None else instance_name
        tf = Matrix4x4.identity() if transform is None else transform
        self._items.append((inst, part, tf))
        return self

    def __len__(self) -> int:
        return len(self._items)

    def parts(self) -> List[Tuple[str, Part]]:
        """Returns list of (instance_name, transformed_part)."""
        res: List[Tuple[str, Part]] = []
        for name, part, tf in self._items:
            res.append((name, part.transform(tf)))
        return res

    @property
    def total_volume(self) -> float:
        """Total volume sum of all parts in mm^3."""
        return sum(part.volume for _, part in self.parts())

    @property
    def total_mass_grams(self) -> float:
        """Total mass of assembly in grams."""
        return sum(part.mass_grams for _, part in self.parts())

    @property
    def center_of_mass(self) -> Vector3D:
        """Mass-weighted assembly center of mass in 3D world space."""
        total_m = self.total_mass_grams
        if total_m <= 1e-9:
            return Vector3D(0, 0, 0)
        wx, wy, wz = 0.0, 0.0, 0.0
        for _, part in self.parts():
            m = part.mass_grams
            c = part.centroid
            wx += c.x * m
            wy += c.y * m
            wz += c.z * m
        return Vector3D(wx / total_m, wy / total_m, wz / total_m)

    @property
    def bounding_box(self) -> BoundingBox3D:
        """Global composite bounding box enclosing all parts."""
        bbox = BoundingBox3D()
        for _, part in self.parts():
            bbox.include_box(part.bounding_box)
        return bbox

    def check_interferences(self, volume_tolerance: float = 1e-3) -> List[Dict[str, Union[str, float]]]:
        """
        Detects 3D geometric collisions / clashing between any pair of parts in the assembly.
        Uses broadphase bounding-box rejection followed by exact CSG boolean intersection.
        """
        clashes: List[Dict[str, Union[str, float]]] = []
        part_list = self.parts()
        n = len(part_list)

        for i in range(n):
            name_a, part_a = part_list[i]
            box_a = part_a.bounding_box

            for j in range(i + 1, n):
                name_b, part_b = part_list[j]
                box_b = part_b.bounding_box

                # Broadphase: AABB intersection test
                if not box_a.intersects(box_b):
                    continue

                # Narrowphase: exact CSG intersection volume
                inter = part_a.intersect(part_b)
                inter_vol = inter.volume
                if inter_vol > volume_tolerance:
                    clashes.append({
                        "part_a": name_a,
                        "part_b": name_b,
                        "clash_volume_mm3": round(inter_vol, 4),
                        "clash_centroid": inter.centroid,
                    })
        return clashes

    def export_composite_obj(self, file_path: str) -> int:
        """
        Exports all parts in the assembly into a single multi-object Wavefront OBJ file.
        Preserves individual object groups ('o PartName').
        """
        lines: List[str] = [
            f"# Solida Multi-Part Assembly: {self.name}",
            f"# Total Parts: {len(self._items)}",
            f"# Total Mass: {self.total_mass_grams:.2f} g",
        ]

        global_vert_offset = 1  # Wavefront OBJ 1-indexed

        for inst_name, part in self.parts():
            lines.append(f"\no {inst_name}")
            facets = tessellate_solid(part.solid)

            v_list: List[Vector3D] = []
            v_map: Dict[Tuple[float, float, float], int] = {}

            def get_vid(p: Vector3D) -> int:
                key = (round(p.x, 6), round(p.y, 6), round(p.z, 6))
                if key not in v_map:
                    idx = len(v_list) + global_vert_offset
                    v_map[key] = idx
                    v_list.append(p)
                    return idx
                return v_map[key]

            f_indices: List[Tuple[int, int, int]] = []
            for f in facets:
                i0 = get_vid(f.v0)
                i1 = get_vid(f.v1)
                i2 = get_vid(f.v2)
                f_indices.append((i0, i1, i2))

            for v in v_list:
                lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")
            for i0, i1, i2 in f_indices:
                lines.append(f"f {i0} {i1} {i2}")

            global_vert_offset += len(v_list)

        text = "\n".join(lines) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return len(text)
