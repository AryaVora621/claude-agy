"""Unit tests for Optical Materials, Sellmeier Dispersion, and Glass Catalogs."""

import unittest

from rayoptix.material import (
    ConstantIndexMaterial,
    MaterialCatalog,
    SellmeierMaterial,
)
from rayoptix.ray import WAVELENGTH_C, WAVELENGTH_D, WAVELENGTH_F


class TestMaterialDispersion(unittest.TestCase):
    """Test suite for optical material refractive indices and Sellmeier models."""

    def test_air_and_vacuum(self) -> None:
        vac = MaterialCatalog.get("VACUUM")
        air = MaterialCatalog.get("AIR")
        self.assertAlmostEqual(vac.refractive_index(500.0), 1.0)
        self.assertAlmostEqual(air.refractive_index(500.0), 1.000277)

    def test_nbk7_dispersion(self) -> None:
        bk7 = MaterialCatalog.get("N-BK7")
        self.assertIsInstance(bk7, SellmeierMaterial)

        # Standard optical properties of Schott N-BK7
        # nd ~ 1.51680, Vd ~ 64.17
        nd = bk7.nd
        vd = bk7.abbe_number
        self.assertAlmostEqual(nd, 1.51680, places=4)
        self.assertAlmostEqual(vd, 64.17, places=1)

        # Normal dispersion: nF (blue 486nm) > nd (yellow 587nm) > nC (red 656nm)
        self.assertGreater(bk7.nF, bk7.nd)
        self.assertGreater(bk7.nd, bk7.nC)
        self.assertEqual(bk7.category, "Crown")

    def test_nsf11_dispersion(self) -> None:
        sf11 = MaterialCatalog.get("N-SF11")
        self.assertIsInstance(sf11, SellmeierMaterial)

        # Standard properties of Schott N-SF11 (dense flint)
        # nd ~ 1.7847, Vd ~ 25.68
        self.assertAlmostEqual(sf11.nd, 1.7847, places=3)
        self.assertAlmostEqual(sf11.abbe_number, 25.68, places=1)
        self.assertEqual(sf11.category, "Flint")
        self.assertLessEqual(sf11.abbe_number, 50.0)

    def test_fused_silica_and_caf2(self) -> None:
        fs = MaterialCatalog.get("FUSED_SILICA")
        caf2 = MaterialCatalog.get("CAF2")

        # Fused silica: nd ~ 1.4585, Vd ~ 67.8
        self.assertAlmostEqual(fs.nd, 1.4585, places=3)
        self.assertAlmostEqual(fs.abbe_number, 67.8, places=1)

        # Fluorite CaF2: ultra-low dispersion Vd ~ 95.0
        self.assertAlmostEqual(caf2.nd, 1.4338, places=3)
        self.assertAlmostEqual(caf2.abbe_number, 95.0, places=1)

    def test_catalog_fallback_and_custom_registration(self) -> None:
        # Fallback on unknown
        fallback = MaterialCatalog.get("NONEXISTENT_CRYSTAL")
        self.assertEqual(fallback.name, "AIR")

        # Custom material
        custom = ConstantIndexMaterial(name="LIQUID_WATER", fixed_index=1.333)
        MaterialCatalog.register(custom)
        retrieved = MaterialCatalog.get("LIQUID_WATER")
        self.assertEqual(retrieved.nd, 1.333)


if __name__ == "__main__":
    unittest.main()
