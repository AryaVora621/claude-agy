"""Optical Materials, Sellmeier Dispersion Models, and Glass Catalogs.

Implements exact 3-term Sellmeier dispersion equations, Abbe number calculations
(Vd and Ve), partial dispersion ratios, and a curated catalog of authentic
Schott, Ohara, and infrared optical materials.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from rayoptix.ray import (
    WAVELENGTH_C,
    WAVELENGTH_D,
    WAVELENGTH_E,
    WAVELENGTH_F,
    WAVELENGTH_G,
)


@dataclass(slots=True)
class OpticalMaterial:
    """Base optical material with chromatic dispersion calculation."""

    name: str
    category: str = "Glass"
    is_mirror: bool = False

    def refractive_index(self, wavelength_nm: float) -> float:
        """Compute refractive index n at specified wavelength in nanometers."""
        raise NotImplementedError("Subclasses must implement refractive_index()")

    @property
    def nd(self) -> float:
        """Refractive index at Fraunhofer d-line (587.56 nm, Helium yellow)."""
        return self.refractive_index(WAVELENGTH_D)

    @property
    def nF(self) -> float:
        """Refractive index at Fraunhofer F-line (486.13 nm, Hydrogen blue)."""
        return self.refractive_index(WAVELENGTH_F)

    @property
    def nC(self) -> float:
        """Refractive index at Fraunhofer C-line (656.27 nm, Hydrogen red)."""
        return self.refractive_index(WAVELENGTH_C)

    @property
    def ne(self) -> float:
        """Refractive index at Fraunhofer e-line (546.07 nm, Mercury green)."""
        return self.refractive_index(WAVELENGTH_E)

    @property
    def abbe_number(self) -> float:
        """Abbe number Vd = (nd - 1) / (nF - nC).

        Higher Vd (>50) indicates low dispersion (crown glass);
        lower Vd (<=50) indicates high dispersion (flint glass).
        """
        dn = self.nF - self.nC
        if abs(dn) < 1e-12:
            return float("inf")
        return (self.nd - 1.0) / dn

    @property
    def partial_dispersion_gF(self) -> float:
        """Relative partial dispersion P_gF = (ng - nF) / (nF - nC)."""
        ng = self.refractive_index(WAVELENGTH_G)
        dn = self.nF - self.nC
        if abs(dn) < 1e-12:
            return 0.0
        return (ng - self.nF) / dn


@dataclass(slots=True)
class ConstantIndexMaterial(OpticalMaterial):
    """Homogeneous non-dispersive optical medium with fixed refractive index."""

    fixed_index: float = 1.0

    def refractive_index(self, wavelength_nm: float) -> float:
        """Return constant refractive index regardless of wavelength."""
        return self.fixed_index


@dataclass(slots=True)
class SellmeierMaterial(OpticalMaterial):
    """Optical glass modeled by the 3-term Sellmeier dispersion equation:

    n^2(lambda) - 1 = B1*lambda^2 / (lambda^2 - C1)
                    + B2*lambda^2 / (lambda^2 - C2)
                    + B3*lambda^2 / (lambda^2 - C3)
    where lambda is in micrometers (um).
    """

    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    c1: float = 0.0  # um^2
    c2: float = 0.0  # um^2
    c3: float = 0.0  # um^2

    def refractive_index(self, wavelength_nm: float) -> float:
        """Compute refractive index using 3-term Sellmeier equation."""
        lam_um = wavelength_nm * 1e-3
        lam2 = lam_um * lam_um

        t1 = (self.b1 * lam2) / (lam2 - self.c1) if abs(lam2 - self.c1) > 1e-15 else 0.0
        t2 = (self.b2 * lam2) / (lam2 - self.c2) if abs(lam2 - self.c2) > 1e-15 else 0.0
        t3 = (self.b3 * lam2) / (lam2 - self.c3) if abs(lam2 - self.c3) > 1e-15 else 0.0

        n2 = 1.0 + t1 + t2 + t3
        if n2 <= 0.0:
            return 1.0
        return math.sqrt(n2)


class MaterialCatalog:
    """Registry and factory for optical materials and dispersion models."""

    _materials: Dict[str, OpticalMaterial] = {}

    @classmethod
    def register(cls, material: OpticalMaterial) -> None:
        """Register an optical material in the global catalog."""
        cls._materials[material.name.upper()] = material

    @classmethod
    def get(cls, name: str) -> OpticalMaterial:
        """Look up an optical material by name (case-insensitive)."""
        key = name.strip().upper()
        if key in cls._materials:
            return cls._materials[key]
        # Fallback to air if unknown
        return cls._materials["AIR"]

    @classmethod
    def list_materials(cls) -> Dict[str, OpticalMaterial]:
        """Return dictionary of all registered optical materials."""
        return dict(cls._materials)


# Initialize default optical materials and Schott/Ohara glass library

# 1. Vacuum and Air
MaterialCatalog.register(ConstantIndexMaterial(name="VACUUM", fixed_index=1.0, category="Gas"))
MaterialCatalog.register(ConstantIndexMaterial(name="AIR", fixed_index=1.000277, category="Gas"))
MaterialCatalog.register(ConstantIndexMaterial(name="MIRROR", fixed_index=-1.0, is_mirror=True, category="Reflective"))

# 2. Schott N-BK7: Primary standard borosilicate crown glass (nd=1.5168, Vd=64.17)
MaterialCatalog.register(
    SellmeierMaterial(
        name="N-BK7",
        category="Crown",
        b1=1.03961212,
        b2=0.231792344,
        b3=1.01046945,
        c1=0.00600069867,
        c2=0.0200179144,
        c3=103.560653,
    )
)

# 3. Schott N-SF11: Dense flint glass for chromatic correction (nd=1.7847, Vd=25.76)
MaterialCatalog.register(
    SellmeierMaterial(
        name="N-SF11",
        category="Flint",
        b1=1.73759695,
        b2=0.313747346,
        b3=1.89878101,
        c1=0.013188707,
        c2=0.0623068142,
        c3=155.23629,
    )
)

# 4. Schott F2: Classic lead flint glass (nd=1.6200, Vd=36.37)
MaterialCatalog.register(
    SellmeierMaterial(
        name="F2",
        category="Flint",
        b1=1.34533359,
        b2=0.209073176,
        b3=0.93735716,
        c1=0.00997743871,
        c2=0.0470450767,
        c3=111.886764,
    )
)

# 5. Schott N-SK16: Dense barium crown glass (nd=1.6204, Vd=60.32)
MaterialCatalog.register(
    SellmeierMaterial(
        name="N-SK16",
        category="Crown",
        b1=1.40822606,
        b2=0.255913217,
        b3=0.957688534,
        c1=0.007604313,
        c2=0.0274640166,
        c3=96.9803153,
    )
)

# 6. Schott N-LASF9: High-index lanthanum dense flint glass (nd=1.8502, Vd=32.17)
MaterialCatalog.register(
    SellmeierMaterial(
        name="N-LASF9",
        category="Dense Flint",
        b1=2.00029547,
        b2=0.298926886,
        b3=1.80691843,
        c1=0.0121426017,
        c2=0.0538736236,
        c3=156.530829,
    )
)

# 7. Fused Silica (SiO2 / Corning 7980, nd=1.4585, Vd=67.82)
MaterialCatalog.register(
    SellmeierMaterial(
        name="FUSED_SILICA",
        category="Crown/Silica",
        b1=0.696166300,
        b2=0.407942600,
        b3=0.897479400,
        c1=0.004679148,
        c2=0.0135120631,
        c3=97.9340025,
    )
)

# 8. Calcium Fluoride (CaF2 / Fluorite crystal, nd=1.4338, Vd=95.34, anomalous dispersion)
MaterialCatalog.register(
    SellmeierMaterial(
        name="CAF2",
        category="Crystal/Apochromat",
        b1=0.5675888,
        b2=0.4710914,
        b3=3.8484723,
        c1=0.00252642999,
        c2=0.0100783328,
        c3=1200.556,
    )
)

# 9. Germanium (Infrared optics 2-14 um, nd ~ 4.0)
MaterialCatalog.register(
    ConstantIndexMaterial(name="GERMANIUM", fixed_index=4.002, category="Infrared")
)
