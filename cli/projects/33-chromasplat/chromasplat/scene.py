"""Scene management, PLY binary/ASCII codecs, and procedural scene generators.

Provides high-level Gaussian scene manipulation, Stanford PLY format serialization
compatible with the Inria 3D Gaussian Splatting standard, and rich procedural
test scenes (Planetary Rings, Cornell Box with SH specular spheres, DNA Double Helix).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from chromasplat.gaussian import Gaussian3D, Quaternion
from chromasplat.spherical_harmonics import create_specular_sh, rgb_to_sh_deg0, sh_deg0_to_rgb


@dataclass
class GaussianScene:
    """Collection of 3D Gaussians representing a volumetric radiance scene.

    Attributes:
        gaussians: List of 3D Gaussian primitives.
        name: Identifying label for telemetry and display.
    """

    gaussians: List[Gaussian3D] = field(default_factory=list)
    name: str = "Scene"

    def __len__(self) -> int:
        return len(self.gaussians)

    @property
    def count(self) -> int:
        """Total number of Gaussians in the scene."""
        return len(self.gaussians)

    def add(self, gaussian: Gaussian3D) -> None:
        """Add a single Gaussian primitive to the scene."""
        self.gaussians.append(gaussian)

    def extend(self, new_gaussians: Sequence[Gaussian3D]) -> None:
        """Extend scene with a sequence of Gaussian primitives."""
        self.gaussians.extend(new_gaussians)

    def get_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute axis-aligned bounding box (min_xyz, max_xyz) across Gaussian centers."""
        if not self.gaussians:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for g in self.gaussians:
            px, py, pz = g.position
            if px < min_x:
                min_x = px
            if py < min_y:
                min_y = py
            if pz < min_z:
                min_z = pz
            if px > max_x:
                max_x = px
            if py > max_y:
                max_y = py
            if pz > max_z:
                max_z = pz

        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

    def get_center(self) -> Tuple[float, float, float]:
        """Compute geometric centroid of Gaussian positions."""
        if not self.gaussians:
            return (0.0, 0.0, 0.0)
        sx = sy = sz = 0.0
        for g in self.gaussians:
            sx += g.position[0]
            sy += g.position[1]
            sz += g.position[2]
        inv = 1.0 / len(self.gaussians)
        return (sx * inv, sy * inv, sz * inv)

    def translate(self, dx: float, dy: float, dz: float) -> GaussianScene:
        """Translate all Gaussians in-place and return self."""
        self.gaussians = [g.translate(dx, dy, dz) for g in self.gaussians]
        return self

    def scale_uniform(self, factor: float) -> GaussianScene:
        """Scale all Gaussians relative to centroid and return self."""
        cx, cy, cz = self.get_center()
        new_list: List[Gaussian3D] = []
        for g in self.gaussians:
            px = cx + (g.position[0] - cx) * factor
            py = cy + (g.position[1] - cy) * factor
            pz = cz + (g.position[2] - cz) * factor
            new_list.append(
                Gaussian3D(
                    position=(px, py, pz),
                    scale=(g.scale[0] * factor, g.scale[1] * factor, g.scale[2] * factor),
                    rotation=g.rotation,
                    opacity=g.opacity,
                    sh_coeffs=g.sh_coeffs,
                )
            )
        self.gaussians = new_list
        return self

    def prune(self, min_opacity: float = 0.01, max_scale: float = 20.0) -> int:
        """Prune nearly transparent or excessively large Gaussians.

        Returns:
            Number of Gaussians removed.
        """
        initial_len = len(self.gaussians)
        self.gaussians = [
            g
            for g in self.gaussians
            if g.opacity >= min_opacity
            and g.scale[0] <= max_scale
            and g.scale[1] <= max_scale
            and g.scale[2] <= max_scale
        ]
        return initial_len - len(self.gaussians)


class PLYCodec:
    """Reader and writer for Stanford PLY 3D Gaussian Splatting files."""

    @staticmethod
    def save_ascii(scene: GaussianScene, filepath: Union[str, Path]) -> None:
        """Save Gaussian scene to ASCII format PLY file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(scene.gaussians)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]

        for g in scene.gaussians:
            px, py, pz = g.position
            # Scale in log space for 3DGS compatibility
            s0 = math.log(max(1e-6, g.scale[0]))
            s1 = math.log(max(1e-6, g.scale[1]))
            s2 = math.log(max(1e-6, g.scale[2]))
            # Opacity in logit space
            op_clamped = max(1e-4, min(1.0 - 1e-4, g.opacity))
            logit_op = math.log(op_clamped / (1.0 - op_clamped))
            # Quaternions: rot_0 is w, rot_1 is x, rot_2 is y, rot_3 is z
            q = g.rotation.normalized()
            # SH degree 0
            sh0 = g.sh_coeffs[0] if g.sh_coeffs else (0.0, 0.0, 0.0)

            line = (
                f"{px:.6f} {py:.6f} {pz:.6f} 0 0 0 "
                f"{sh0[0]:.6f} {sh0[1]:.6f} {sh0[2]:.6f} "
                f"{logit_op:.6f} {s0:.6f} {s1:.6f} {s2:.6f} "
                f"{q.w:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f}"
            )
            lines.append(line)

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def load_ascii(filepath: Union[str, Path]) -> GaussianScene:
        """Load Gaussian scene from ASCII format PLY file."""
        path = Path(filepath)
        lines = path.read_text(encoding="utf-8").splitlines()

        header_ended = False
        num_vertices = 0
        gaussians: List[Gaussian3D] = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith("comment"):
                continue

            if not header_ended:
                parts = line.split()
                if parts[0] == "element" and parts[1] == "vertex":
                    num_vertices = int(parts[2])
                elif parts[0] == "end_header":
                    header_ended = True
                continue

            tokens = line.split()
            if len(tokens) < 17:
                continue

            px, py, pz = float(tokens[0]), float(tokens[1]), float(tokens[2])
            # tokens[3..5] are normals nx, ny, nz
            f_dc_0, f_dc_1, f_dc_2 = float(tokens[6]), float(tokens[7]), float(tokens[8])
            logit_op = float(tokens[9])
            s0, s1, s2 = float(tokens[10]), float(tokens[11]), float(tokens[12])
            rw, rx, ry, rz = float(tokens[13]), float(tokens[14]), float(tokens[15]), float(tokens[16])

            # Invert log scale and logit opacity
            scale = (math.exp(s0), math.exp(s1), math.exp(s2))
            opacity = 1.0 / (1.0 + math.exp(-logit_op))
            rot = Quaternion(rw, rx, ry, rz).normalized()
            sh = ((f_dc_0, f_dc_1, f_dc_2),)

            gaussians.append(
                Gaussian3D(
                    position=(px, py, pz),
                    scale=scale,
                    rotation=rot,
                    opacity=opacity,
                    sh_coeffs=sh,
                )
            )

        return GaussianScene(gaussians=gaussians, name=path.stem)


class SceneFactory:
    """Procedural generation of complex 3D Gaussian Splatting benchmark scenes."""

    @staticmethod
    def orbiting_rings(
        num_planet_splats: int = 120,
        num_ring_splats: int = 360,
    ) -> GaussianScene:
        """Construct glowing gas giant planet with illuminated concentric rings.

        Args:
            num_planet_splats: Number of Gaussians forming the central planet sphere.
            num_ring_splats: Number of Gaussians forming the flat orbital rings.

        Returns:
            GaussianScene containing planet and ring system.
        """
        gaussians: List[Gaussian3D] = []

        # 1. Central Planet Sphere (Golden/Bronze Jovian Bands)
        radius_planet = 0.8
        phi_golden = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio for Fibonacci sphere

        for i in range(num_planet_splats):
            y = 1.0 - (i / float(num_planet_splats - 1)) * 2.0  # y in [-1, 1]
            radius_at_y = math.sqrt(max(0.0, 1.0 - y * y))
            theta = 2.0 * math.pi * i / phi_golden

            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y

            px = x * radius_planet
            py = y * radius_planet
            pz = z * radius_planet

            # Latitude-based atmospheric banding (Jupiter/Saturn stripes)
            band = math.sin(py * 8.0)
            if band > 0.3:
                r, g, b = 0.95, 0.65, 0.35  # Warm amber band
            elif band < -0.3:
                r, g, b = 0.85, 0.45, 0.20  # Deep rust band
            else:
                r, g, b = 0.92, 0.82, 0.62  # Cream haze band

            # Normal vector for specular highlight
            norm_x, norm_y, norm_z = x, y, z
            sh = create_specular_sh(
                base_color=(r, g, b),
                highlight_color=(1.0, 0.95, 0.85),
                specular_dir=(norm_x, norm_y, norm_z),
                degree=2,
                shininess=0.4,
            )

            # Oriented tangential Gaussian splat
            rot = Quaternion.from_axis_angle((x, y, z), theta)
            g = Gaussian3D(
                position=(px, py, pz),
                scale=(0.14, 0.14, 0.14),
                rotation=rot,
                opacity=0.92,
                sh_coeffs=sh,
            )
            gaussians.append(g)

        # 2. Concentric Orbital Rings (Tilted by 25 degrees)
        ring_tilt = math.radians(25.0)
        cos_tilt = math.cos(ring_tilt)
        sin_tilt = math.sin(ring_tilt)

        for i in range(num_ring_splats):
            # Radial distribution between r_inner = 1.2 and r_outer = 2.4
            frac = i / float(num_ring_splats)
            r_ring = 1.25 + 1.15 * math.sqrt(frac)
            angle = frac * 14.0 * math.pi + (i % 7) * 0.2

            # Ring gap (Cassini Division equivalent)
            if 1.70 <= r_ring <= 1.85:
                continue

            rx = r_ring * math.cos(angle)
            rz = r_ring * math.sin(angle)
            # Tilt plane: y = rz * sin(tilt), z = rz * cos(tilt)
            px = rx
            py = rz * sin_tilt
            pz = rz * cos_tilt

            # Optical color gradient across ring radius
            ring_brightness = 0.5 + 0.45 * math.sin(r_ring * 12.0)
            if r_ring < 1.7:
                cr, cg, cb = 0.85 * ring_brightness, 0.75 * ring_brightness, 0.55 * ring_brightness
            else:
                cr, cg, cb = 0.60 * ring_brightness, 0.70 * ring_brightness, 0.90 * ring_brightness

            sh0 = rgb_to_sh_deg0((cr, cg, cb))
            # Thin disc-like splats flattened along normal
            q_ring = Quaternion.from_axis_angle((1.0, 0.0, 0.0), ring_tilt)
            g_ring = Gaussian3D(
                position=(px, py, pz),
                scale=(0.10, 0.02, 0.10),
                rotation=q_ring,
                opacity=0.85,
                sh_coeffs=(sh0,),
            )
            gaussians.append(g_ring)

        return GaussianScene(gaussians=gaussians, name="SaturnianRings")

    @staticmethod
    def cornell_box() -> GaussianScene:
        """Construct Cornell Box benchmark with diffuse walls and central specular sphere."""
        gaussians: List[Gaussian3D] = []

        # Box dimensions: [-1.2, 1.2] on each axis
        # Back wall (White)
        for y_i in range(-5, 6):
            for x_i in range(-5, 6):
                px = x_i * 0.22
                py = y_i * 0.22
                pz = -1.2
                sh = (rgb_to_sh_deg0((0.85, 0.85, 0.85)),)
                gaussians.append(
                    Gaussian3D(
                        position=(px, py, pz),
                        scale=(0.15, 0.15, 0.04),
                        opacity=0.98,
                        sh_coeffs=sh,
                    )
                )

        # Left wall (Red)
        for y_i in range(-5, 6):
            for z_i in range(-5, 6):
                px = -1.2
                py = y_i * 0.22
                pz = z_i * 0.22
                sh = (rgb_to_sh_deg0((0.92, 0.15, 0.15)),)
                gaussians.append(
                    Gaussian3D(
                        position=(px, py, pz),
                        scale=(0.04, 0.15, 0.15),
                        opacity=0.98,
                        sh_coeffs=sh,
                    )
                )

        # Right wall (Green)
        for y_i in range(-5, 6):
            for z_i in range(-5, 6):
                px = 1.2
                py = y_i * 0.22
                pz = z_i * 0.22
                sh = (rgb_to_sh_deg0((0.15, 0.90, 0.20)),)
                gaussians.append(
                    Gaussian3D(
                        position=(px, py, pz),
                        scale=(0.04, 0.15, 0.15),
                        opacity=0.98,
                        sh_coeffs=sh,
                    )
                )

        # Floor (White)
        for x_i in range(-5, 6):
            for z_i in range(-5, 6):
                px = x_i * 0.22
                py = -1.2
                pz = z_i * 0.22
                sh = (rgb_to_sh_deg0((0.88, 0.88, 0.88)),)
                gaussians.append(
                    Gaussian3D(
                        position=(px, py, pz),
                        scale=(0.15, 0.04, 0.15),
                        opacity=0.98,
                        sh_coeffs=sh,
                    )
                )

        # Ceiling (White)
        for x_i in range(-5, 6):
            for z_i in range(-5, 6):
                px = x_i * 0.22
                py = 1.2
                pz = z_i * 0.22
                sh = (rgb_to_sh_deg0((0.88, 0.88, 0.88)),)
                gaussians.append(
                    Gaussian3D(
                        position=(px, py, pz),
                        scale=(0.15, 0.04, 0.15),
                        opacity=0.98,
                        sh_coeffs=sh,
                    )
                )

        # Center Sphere with Degree-2 Spherical Harmonics Specular Highlight
        sphere_center = (0.0, -0.4, 0.0)
        sphere_radius = 0.55
        n_pts = 90
        phi_golden = (1.0 + math.sqrt(5.0)) / 2.0

        for i in range(n_pts):
            y = 1.0 - (i / float(n_pts - 1)) * 2.0
            r_y = math.sqrt(max(0.0, 1.0 - y * y))
            theta = 2.0 * math.pi * i / phi_golden
            x = math.cos(theta) * r_y
            z = math.sin(theta) * r_y

            px = sphere_center[0] + x * sphere_radius
            py = sphere_center[1] + y * sphere_radius
            pz = sphere_center[2] + z * sphere_radius

            # Specular highlight shining in direction (0.4, 0.8, 0.5)
            sh_coeffs = create_specular_sh(
                base_color=(0.15, 0.45, 0.95),  # Metallic blue base
                highlight_color=(1.0, 1.0, 1.0),  # Pure specular glint
                specular_dir=(x, y, z),
                degree=2,
                shininess=0.8,
            )

            gaussians.append(
                Gaussian3D(
                    position=(px, py, pz),
                    scale=(0.12, 0.12, 0.12),
                    opacity=0.99,
                    sh_coeffs=sh_coeffs,
                )
            )

        return GaussianScene(gaussians=gaussians, name="CornellBox")

    @staticmethod
    def dna_double_helix(
        num_turns: float = 2.5,
        total_height: float = 3.0,
        radius: float = 0.75,
        samples_per_turn: int = 40,
    ) -> GaussianScene:
        """Construct glowing DNA double helix with complementary base pairs.

        Args:
            num_turns: Number of complete 360-degree helical revolutions.
            total_height: Vertical extent of the DNA strand.
            radius: Radius of the helical cylinders.
            samples_per_turn: Discretization density per turn.

        Returns:
            GaussianScene containing DNA strand.
        """
        gaussians: List[Gaussian3D] = []
        total_samples = int(num_turns * samples_per_turn)

        for i in range(total_samples):
            t = i / float(total_samples - 1)
            y = (t - 0.5) * total_height
            angle1 = t * num_turns * 2.0 * math.pi
            angle2 = angle1 + math.pi  # Opposite strand

            # Strand 1 (Cyan-Blue Backbone)
            x1 = radius * math.cos(angle1)
            z1 = radius * math.sin(angle1)
            sh1 = (rgb_to_sh_deg0((0.15, 0.80, 0.95)),)
            gaussians.append(
                Gaussian3D(
                    position=(x1, y, z1),
                    scale=(0.08, 0.08, 0.08),
                    opacity=0.95,
                    sh_coeffs=sh1,
                )
            )

            # Strand 2 (Magenta-Gold Backbone)
            x2 = radius * math.cos(angle2)
            z2 = radius * math.sin(angle2)
            sh2 = (rgb_to_sh_deg0((0.95, 0.30, 0.75)),)
            gaussians.append(
                Gaussian3D(
                    position=(x2, y, z2),
                    scale=(0.08, 0.08, 0.08),
                    opacity=0.95,
                    sh_coeffs=sh2,
                )
            )

            # Base Pair Bridges (Every 2 steps)
            if i % 2 == 0:
                # 4 intermediate splats bridging Strand 1 and Strand 2
                for s in (0.25, 0.50, 0.75):
                    bx = x1 + s * (x2 - x1)
                    bz = z1 + s * (z2 - z1)

                    # Alternating A-T (amber) and G-C (emerald) base pairs
                    if (i // 2) % 2 == 0:
                        pair_color = (0.95, 0.75, 0.20)  # Adenine - Thymine
                    else:
                        pair_color = (0.20, 0.90, 0.40)  # Guanine - Cytosine

                    sh_pair = (rgb_to_sh_deg0(pair_color),)
                    gaussians.append(
                        Gaussian3D(
                            position=(bx, y, bz),
                            scale=(0.05, 0.05, 0.05),
                            opacity=0.85,
                            sh_coeffs=sh_pair,
                        )
                    )

        return GaussianScene(gaussians=gaussians, name="DNADoubleHelix")
