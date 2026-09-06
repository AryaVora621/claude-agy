"""
Curated Library of 3D Molecular Structures for SpectroChem 3D.
Contains precise 3D Cartesian coordinates, atomic partial charges, and connectivity.
"""

from typing import Dict, Callable

try:
    from .chem_engine import Molecule
except ImportError:
    from chem_engine import Molecule


def build_water() -> Molecule:
    """Water (H2O): C2v bent geometry, 104.5 degree bond angle."""
    mol = Molecule("Water (H2O)")
    # Central Oxygen
    mol.add_atom("O", 0.000, 0.000, 0.117, charge=-0.834, name="O1")
    # Hydrogens
    mol.add_atom("H", 0.000, 0.757, -0.469, charge=0.417, name="H1")
    mol.add_atom("H", 0.000, -0.757, -0.469, charge=0.417, name="H2")
    mol.detect_connectivity()
    return mol


def build_carbon_dioxide() -> Molecule:
    """Carbon Dioxide (CO2): D_inf_h linear geometry, 180 degree angle."""
    mol = Molecule("Carbon Dioxide (CO2)")
    mol.add_atom("C", 0.000, 0.000, 0.000, charge=0.650, name="C1")
    mol.add_atom("O", 1.160, 0.000, 0.000, charge=-0.325, name="O1")
    mol.add_atom("O", -1.160, 0.000, 0.000, charge=-0.325, name="O2")
    mol.detect_connectivity()
    return mol


def build_methane() -> Molecule:
    """Methane (CH4): Td tetrahedral symmetry, 109.47 degree bond angles."""
    mol = Molecule("Methane (CH4)")
    mol.add_atom("C", 0.000, 0.000, 0.000, charge=-0.440, name="C1")
    d = 0.629
    mol.add_atom("H",  d,  d,  d, charge=0.110, name="H1")
    mol.add_atom("H", -d, -d,  d, charge=0.110, name="H2")
    mol.add_atom("H", -d,  d, -d, charge=0.110, name="H3")
    mol.add_atom("H",  d, -d, -d, charge=0.110, name="H4")
    mol.detect_connectivity()
    return mol


def build_ammonia() -> Molecule:
    """Ammonia (NH3): C3v trigonal pyramidal geometry with lone pair umbrella."""
    mol = Molecule("Ammonia (NH3)")
    mol.add_atom("N", 0.000, 0.000, 0.116, charge=-0.750, name="N1")
    mol.add_atom("H", 0.000, 0.940, -0.270, charge=0.250, name="H1")
    mol.add_atom("H", 0.814, -0.470, -0.270, charge=0.250, name="H2")
    mol.add_atom("H", -0.814, -0.470, -0.270, charge=0.250, name="H3")
    mol.detect_connectivity()
    return mol


def build_benzene() -> Molecule:
    """Benzene (C6H6): D6h planar aromatic ring with delocalized pi electrons."""
    mol = Molecule("Benzene (C6H6)")
    rc = 1.395  # Carbon ring radius (Angstroms)
    rh = 2.475  # Hydrogen ring radius (Angstroms)
    import math

    for i in range(6):
        angle = i * (math.pi / 3.0)
        cx = rc * math.cos(angle)
        cy = rc * math.sin(angle)
        hx = rh * math.cos(angle)
        hy = rh * math.sin(angle)
        mol.add_atom("C", cx, cy, 0.0, charge=-0.120, name=f"C{i+1}")
        mol.add_atom("H", hx, hy, 0.0, charge=0.120, name=f"H{i+1}")

    mol.detect_connectivity()
    return mol


def build_ethanol() -> Molecule:
    """Ethanol (C2H5OH): Aliphatic alcohol with C-C, C-O, and O-H bonds."""
    mol = Molecule("Ethanol (C2H5OH)")
    # Carbon backbone
    mol.add_atom("C", -0.665, -0.272,  0.000, charge=-0.200, name="C1")
    mol.add_atom("C",  0.722,  0.371,  0.000, charge=0.100,  name="C2")
    mol.add_atom("O",  1.670, -0.686,  0.000, charge=-0.600, name="O1")

    # Methyl Hydrogens
    mol.add_atom("H", -0.762, -0.912,  0.884, charge=0.060, name="H1")
    mol.add_atom("H", -0.762, -0.912, -0.884, charge=0.060, name="H2")
    mol.add_atom("H", -1.455,  0.485,  0.000, charge=0.060, name="H3")

    # Methylene Hydrogens
    mol.add_atom("H",  0.840,  1.002,  0.889, charge=0.060, name="H4")
    mol.add_atom("H",  0.840,  1.002, -0.889, charge=0.060, name="H5")

    # Hydroxyl Hydrogen
    mol.add_atom("H",  2.551, -0.301,  0.000, charge=0.400, name="H6")

    mol.detect_connectivity()
    return mol


def build_caffeine() -> Molecule:
    """Caffeine (C8H10N4O2): Fused purine bicyclic ring alkaloid."""
    mol = Molecule("Caffeine (C8H10N4O2)")
    # Ring atoms
    mol.add_atom("N", -0.85,  1.32,  0.00, charge=-0.35, name="N1")
    mol.add_atom("C",  0.48,  1.65,  0.00, charge=0.45,  name="C2")
    mol.add_atom("O",  0.85,  2.81,  0.00, charge=-0.50, name="O2")
    mol.add_atom("N",  1.38,  0.60,  0.00, charge=-0.35, name="N3")
    mol.add_atom("C",  0.92, -0.73,  0.00, charge=0.40,  name="C4")
    mol.add_atom("C", -0.45, -1.02,  0.00, charge=0.10,  name="C5")
    mol.add_atom("C", -1.38,  0.05,  0.00, charge=0.50,  name="C6")
    mol.add_atom("O", -2.60, -0.09,  0.00, charge=-0.50, name="O6")

    # Imidazole ring
    mol.add_atom("N", -0.56, -2.39,  0.00, charge=-0.30, name="N7")
    mol.add_atom("C",  0.69, -2.85,  0.00, charge=0.30,  name="C8")
    mol.add_atom("H",  0.95, -3.90,  0.00, charge=0.15,  name="H8")
    mol.add_atom("N",  1.61, -1.89,  0.00, charge=-0.30, name="N9")

    # Methyl groups
    # N1 methyl
    mol.add_atom("C", -1.85,  2.37,  0.00, charge=0.10,  name="C10")
    mol.add_atom("H", -1.45,  3.38,  0.00, charge=0.08,  name="H10a")
    mol.add_atom("H", -2.48,  2.24,  0.89, charge=0.08,  name="H10b")
    mol.add_atom("H", -2.48,  2.24, -0.89, charge=0.08,  name="H10c")

    # N3 methyl
    mol.add_atom("C",  2.82,  0.90,  0.00, charge=0.10,  name="C11")
    mol.add_atom("H",  3.04,  1.96,  0.00, charge=0.08,  name="H11a")
    mol.add_atom("H",  3.28,  0.44,  0.89, charge=0.08,  name="H11b")
    mol.add_atom("H",  3.28,  0.44, -0.89, charge=0.08,  name="H11c")

    # N7 methyl
    mol.add_atom("C", -1.78, -3.17,  0.00, charge=0.10,  name="C12")
    mol.add_atom("H", -1.54, -4.23,  0.00, charge=0.08,  name="H12a")
    mol.add_atom("H", -2.37, -2.93,  0.89, charge=0.08,  name="H12b")
    mol.add_atom("H", -2.37, -2.93, -0.89, charge=0.08,  name="H12c")

    mol.detect_connectivity()
    return mol


def build_aspirin() -> Molecule:
    """Aspirin / Acetylsalicylic Acid (C9H8O4): Carboxylic acid + ester aromatic."""
    mol = Molecule("Aspirin (C9H8O4)")
    import math

    # Aromatic ring C1 to C6
    rc = 1.40
    for i in range(6):
        a = i * (math.pi / 3.0)
        mol.add_atom("C", rc * math.cos(a), rc * math.sin(a), 0.0, charge=-0.10, name=f"C{i+1}")

    # Ring Hydrogens on C3, C4, C5, C6
    for i in [2, 3, 4, 5]:
        a = i * (math.pi / 3.0)
        mol.add_atom("H", 2.48 * math.cos(a), 2.48 * math.sin(a), 0.0, charge=0.10, name=f"H{i+1}")

    # Carboxylic acid on C1
    mol.add_atom("C",  2.50,  0.00,  0.00, charge=0.55,  name="C7")
    mol.add_atom("O",  3.15,  1.02,  0.00, charge=-0.50, name="O1")
    mol.add_atom("O",  3.05, -1.18,  0.00, charge=-0.55, name="O2")
    mol.add_atom("H",  4.00, -1.10,  0.00, charge=0.45,  name="H7")

    # Acetoxy ester group on C2
    c2x, c2y = rc * math.cos(math.pi / 3.0), rc * math.sin(math.pi / 3.0)
    mol.add_atom("O", c2x + 1.10, c2y + 0.65,  0.00, charge=-0.45, name="O3")
    mol.add_atom("C", c2x + 2.30, c2y + 0.15,  0.00, charge=0.60,  name="C8")
    mol.add_atom("O", c2x + 2.65, c2y - 1.00,  0.00, charge=-0.50, name="O4")
    mol.add_atom("C", c2x + 3.25, c2y + 1.30,  0.00, charge=-0.20, name="C9")

    # Acetyl methyl hydrogens
    mol.add_atom("H", c2x + 4.28, c2y + 0.95,  0.00, charge=0.08,  name="H9a")
    mol.add_atom("H", c2x + 3.12, c2y + 1.93,  0.89, charge=0.08,  name="H9b")
    mol.add_atom("H", c2x + 3.12, c2y + 1.93, -0.89, charge=0.08,  name="H9c")

    mol.detect_connectivity()
    return mol


def build_sulfur_hexafluoride() -> Molecule:
    """Sulfur Hexafluoride (SF6): Oh octahedral symmetry."""
    mol = Molecule("Sulfur Hexafluoride (SF6)")
    mol.add_atom("S", 0.00, 0.00, 0.00, charge=1.80, name="S1")
    d = 1.56
    mol.add_atom("F",  d, 0.0, 0.0, charge=-0.30, name="F1")
    mol.add_atom("F", -d, 0.0, 0.0, charge=-0.30, name="F2")
    mol.add_atom("F", 0.0,  d, 0.0, charge=-0.30, name="F3")
    mol.add_atom("F", 0.0, -d, 0.0, charge=-0.30, name="F4")
    mol.add_atom("F", 0.0, 0.0,  d, charge=-0.30, name="F5")
    mol.add_atom("F", 0.0, 0.0, -d, charge=-0.30, name="F6")
    mol.detect_connectivity()
    return mol


MOLECULE_PRESETS: Dict[str, Callable[[], Molecule]] = {
    "water": build_water,
    "carbon_dioxide": build_carbon_dioxide,
    "methane": build_methane,
    "ammonia": build_ammonia,
    "benzene": build_benzene,
    "ethanol": build_ethanol,
    "caffeine": build_caffeine,
    "aspirin": build_aspirin,
    "sf6": build_sulfur_hexafluoride
}
