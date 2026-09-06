"""Real Spherical Harmonics engine for view-dependent directional radiance.

Implements exact mathematical formulation of real spherical harmonics basis
functions Y_l^m up to degree 3 (16 basis functions) and directional RGB color
evaluation following the 3D Gaussian Splatting standard.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# Exact normalization constants for real spherical harmonics
# Degree 0 (l=0)
C0 = 0.28209479177387814  # 0.5 * sqrt(1/pi)

# Degree 1 (l=1)
C1 = 0.4886025119029199  # 0.5 * sqrt(3/pi)

# Degree 2 (l=2)
C2_0 = 1.0925484305920792  # 0.5 * sqrt(15/pi)
C2_1 = 0.31539156525252005  # 0.25 * sqrt(5/pi)
C2_2 = 0.5462742152960396  # 0.25 * sqrt(15/pi)

# Degree 3 (l=3)
C3_0 = 0.5900435899266435  # 0.25 * sqrt(35 / (2*pi))
C3_1 = 2.890611442640554  # 0.5 * sqrt(105 / pi)
C3_2 = 0.4570457994644658  # 0.25 * sqrt(21 / (2*pi))
C3_3 = 0.3731763325901154  # 0.25 * sqrt(7 / pi)
C3_4 = 1.445305721320277  # 0.25 * sqrt(105 / pi)


def eval_sh_basis(degree: int, dir_vector: Tuple[float, float, float]) -> List[float]:
    """Compute real spherical harmonics basis values Y_l^m for normalized direction (x, y, z).

    Args:
        degree: Maximum spherical harmonics degree in {0, 1, 2, 3}.
        dir_vector: (x, y, z) viewing direction. Normalized internally.

    Returns:
        List of basis evaluations of length (degree + 1)^2.
    """
    deg = max(0, min(3, degree))
    x, y, z = dir_vector
    norm = math.sqrt(x * x + y * y + z * z)
    if norm > 1e-12:
        inv = 1.0 / norm
        x, y, z = x * inv, y * inv, z * inv
    else:
        x, y, z = 0.0, 0.0, 1.0

    basis: List[float] = [C0]
    if deg == 0:
        return basis

    # Degree 1 (l=1, 3 basis functions)
    basis.append(-C1 * y)
    basis.append(C1 * z)
    basis.append(-C1 * x)
    if deg == 1:
        return basis

    # Degree 2 (l=2, 5 basis functions)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    basis.append(C2_0 * xy)
    basis.append(-C2_0 * yz)
    basis.append(C2_1 * (2.0 * zz - xx - yy))
    basis.append(-C2_0 * xz)
    basis.append(C2_2 * (xx - yy))
    if deg == 2:
        return basis

    # Degree 3 (l=3, 7 basis functions)
    basis.append(-C3_0 * y * (3.0 * xx - yy))
    basis.append(C3_1 * xy * z)
    basis.append(-C3_2 * y * (4.0 * zz - xx - yy))
    basis.append(C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
    basis.append(-C3_2 * x * (4.0 * zz - xx - yy))
    basis.append(C3_4 * z * (xx - yy))
    basis.append(-C3_0 * x * (xx - 3.0 * yy))

    return basis


def eval_sh(
    degree: int,
    sh_coeffs: Sequence[Tuple[float, float, float]],
    dir_vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Evaluate view-dependent RGB radiance from spherical harmonics coefficients.

    Args:
        degree: Maximum degree to evaluate (0, 1, 2, or 3).
        sh_coeffs: Sequence of (r, g, b) coefficient tuples of length at least (degree+1)^2.
        dir_vector: Viewing direction vector from point towards camera.

    Returns:
        (r, g, b) tuple clamped to [0.0, 1.0].
    """
    basis = eval_sh_basis(degree, dir_vector)
    num_coeffs = min(len(basis), len(sh_coeffs))

    r = 0.5
    g = 0.5
    b = 0.5

    for i in range(num_coeffs):
        b_val = basis[i]
        c = sh_coeffs[i]
        r += c[0] * b_val
        g += c[1] * b_val
        b += c[2] * b_val

    # Clamp to standard physical RGB dynamic range
    clamped_r = max(0.0, min(1.0, r))
    clamped_g = max(0.0, min(1.0, g))
    clamped_b = max(0.0, min(1.0, b))

    return (clamped_r, clamped_g, clamped_b)


def rgb_to_sh_deg0(color_rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert base RGB color in [0.0, 1.0] to degree-0 spherical harmonics coefficient."""
    inv_c0 = 1.0 / C0
    return (
        (color_rgb[0] - 0.5) * inv_c0,
        (color_rgb[1] - 0.5) * inv_c0,
        (color_rgb[2] - 0.5) * inv_c0,
    )


def sh_deg0_to_rgb(sh0: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert degree-0 spherical harmonics coefficient to base RGB color in [0.0, 1.0]."""
    r = max(0.0, min(1.0, sh0[0] * C0 + 0.5))
    g = max(0.0, min(1.0, sh0[1] * C0 + 0.5))
    b = max(0.0, min(1.0, sh0[2] * C0 + 0.5))
    return (r, g, b)


def create_specular_sh(
    base_color: Tuple[float, float, float],
    highlight_color: Tuple[float, float, float],
    specular_dir: Tuple[float, float, float],
    degree: int = 2,
    shininess: float = 0.6,
) -> Tuple[Tuple[float, float, float], ...]:
    """Synthesize spherical harmonics coefficients exhibiting directional specular highlight.

    Args:
        base_color: Diffuse albedo color (r, g, b) in [0.0, 1.0].
        highlight_color: Specular peak color (r, g, b) in [0.0, 1.0].
        specular_dir: Normal or dominant reflection direction vector.
        degree: Maximum degree (1, 2, or 3).
        shininess: Specular intensity weight in [0.0, 1.0].

    Returns:
        Tuple of (r, g, b) SH coefficient tuples of length (degree + 1)^2.
    """
    deg = max(0, min(3, degree))
    num_coeffs = (deg + 1) * (deg + 1)
    coeffs: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * num_coeffs

    # Degree 0: Base diffuse color
    coeffs[0] = rgb_to_sh_deg0(base_color)

    if deg == 0:
        return tuple(coeffs)

    # Normalize specular direction
    dx, dy, dz = specular_dir
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm > 1e-12:
        dx, dy, dz = dx / norm, dy / norm, dz / norm
    else:
        dx, dy, dz = 0.0, 0.0, 1.0

    # Project directional lobe onto degree 1 harmonics
    # Y_1^-1 (y), Y_1^0 (z), Y_1^1 (x)
    delta_r = (highlight_color[0] - base_color[0]) * shininess
    delta_g = (highlight_color[1] - base_color[1]) * shininess
    delta_b = (highlight_color[2] - base_color[2]) * shininess

    inv_c1 = 1.0 / C1
    # Note basis signs: Y_1^-1 = -C1*y, Y_1^0 = C1*z, Y_1^1 = -C1*x
    weight = 0.5 * inv_c1
    coeffs[1] = (-dy * delta_r * weight, -dy * delta_g * weight, -dy * delta_b * weight)
    coeffs[2] = (dz * delta_r * weight, dz * delta_g * weight, dz * delta_b * weight)
    coeffs[3] = (-dx * delta_r * weight, -dx * delta_g * weight, -dx * delta_b * weight)

    if deg >= 2:
        # Degree 2 adds quadratic directional peaking
        w2 = 0.25 * shininess
        coeffs[4] = (dx * dy * delta_r * w2, dx * dy * delta_g * w2, dx * dy * delta_b * w2)
        coeffs[5] = (-dy * dz * delta_r * w2, -dy * dz * delta_g * w2, -dy * dz * delta_b * w2)
        coeffs[6] = (
            (2.0 * dz * dz - dx * dx - dy * dy) * delta_r * w2,
            (2.0 * dz * dz - dx * dx - dy * dy) * delta_g * w2,
            (2.0 * dz * dz - dx * dx - dy * dy) * delta_b * w2,
        )
        coeffs[7] = (-dx * dz * delta_r * w2, -dx * dz * delta_g * w2, -dx * dz * delta_b * w2)
        coeffs[8] = (
            (dx * dx - dy * dy) * delta_r * w2,
            (dx * dx - dy * dy) * delta_g * w2,
            (dx * dx - dy * dy) * delta_b * w2,
        )

    return tuple(coeffs)
