"""
Interactive Rocket Propulsion & Supersonic Aerodynamics Workbench.

Demonstrates:
  1. De Laval Supersonic Nozzle Synthesis via 2D Method of Characteristics (MOC)
  2. Rocket Engine Thermochemistry & Flight Ascent Performance (Sea Level to Vacuum)
  3. Plume Adaptation Regimes (Summerfield Separation, Overexpanded, Adapted, Underexpanded)
  4. Bartz Convective Heat Transfer & Coupled Regenerative Thrust Chamber Cooling
  5. Sub-Pixel Unicode Braille 2D Visualization with 24-bit TrueColor Telemetry HUD
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermoprop.cooling import (
    RegenerativeCoolingEngine,
    WallMaterial,
)
from thermoprop.gas_dynamics import GasProperties
from thermoprop.moc_nozzle import MethodOfCharacteristicsNozzle
from thermoprop.plume import ExhaustPlumeEngine
from thermoprop.propulsion import (
    PropellantData,
    PropellantLibrary,
    RocketPropulsionEngine,
    SEA_LEVEL_PRESSURE,
)
from thermoprop.visualizer import (
    BrailleCanvas,
    RocketVisualizer,
)


def get_ambient_pressure_at_altitude(altitude_km: float) -> float:
    """
    Standard atmospheric pressure model up to 50 km:
      Troposphere (0 - 11 km): P = P0 * (1 - 0.0065 * h / 288.15) ^ 5.2561
      Stratosphere (11 - 25 km): Isothermal exponential decay
      Above 45 km: Near vacuum
    """
    if altitude_km <= 0.0:
        return SEA_LEVEL_PRESSURE
    if altitude_km >= 45.0:
        return 1.0  # Near vacuum

    h_m = altitude_km * 1000.0
    if h_m <= 11000.0:
        t_b = 288.15
        p_b = 101325.0
        l_b = -0.0065
        return p_b * ((1.0 + (l_b * h_m) / t_b) ** 5.25588)
    elif h_m <= 25000.0:
        # Isothermal layer at 216.65 K
        p_11 = 22632.1
        h_diff = h_m - 11000.0
        return p_11 * math.exp(-h_diff / 6341.6)
    else:
        # Upper stratosphere
        p_25 = 2511.0
        h_diff = h_m - 25000.0
        return max(1.0, p_25 * math.exp(-h_diff / 7000.0))


def run_ascent_mission_demo(
    propellant_name: str = "methalox",
    chamber_pressure_bar: float = 100.0,
    throat_radius_m: float = 0.10,
    expansion_ratio: float = 40.0,
) -> None:
    """Simulate atmospheric ascent through all plume regimes."""
    propellants = {
        "hydrolox": PropellantLibrary.hydrolox(),
        "methalox": PropellantLibrary.methalox(),
        "kerolox": PropellantLibrary.kerolox(),
        "hypergolic": PropellantLibrary.hypergolic(),
    }
    prop = propellants.get(propellant_name.lower(), PropellantLibrary.methalox())
    engine = RocketPropulsionEngine(prop)
    pc = chamber_pressure_bar * 1.0e5

    state = engine.evaluate_engine(
        chamber_pressure=pc,
        throat_radius=throat_radius_m,
        expansion_ratio=expansion_ratio,
    )

    moc = MethodOfCharacteristicsNozzle(prop.gas)
    contour = moc.design_minimum_length_nozzle(
        target_exit_mach=state.exit_mach,
        throat_height=throat_radius_m,
        num_expansion_waves=8,
    )

    cooling = RegenerativeCoolingEngine(state, prop.gas, WallMaterial.copper_cucrzr())
    plume_engine = ExhaustPlumeEngine(prop.gas)

    print("\n" + "=" * 76)
    print(f"   THERMOPROP MISSION LABORATORY: {prop.name.upper()} ROCKET PROPULSION ENGINE")
    print("=" * 76)
    print(f" Thrust Chamber Pressure : {chamber_pressure_bar:6.1f} bar")
    print(f" Throat Diameter         : {2.0 * throat_radius_m * 1000.0:6.1f} mm")
    print(f" Expansion Ratio A_e/A_t : {expansion_ratio:6.1f}")
    print(f" Exit Mach Number        : {state.exit_mach:6.2f}")
    print(f" Vacuum Thrust           : {state.thrust_vacuum / 1e3:6.1f} kN")
    print(f" Vacuum Specific Impulse : {state.isp_vacuum_seconds:6.1f} s")
    print(f" Sea-Level Specific Imp. : {state.isp_sea_level_seconds:6.1f} s")

    # Ascent profile stations: (Altitude in km, Title)
    stations = [
        (0.0, "SEA-LEVEL LIFTOFF (OVEREXPANDED WITH LIP OBLIQUE SHOCKS)"),
        (8.0, "TROPOSPHERE 8 KM (PERFECTLY EXPANDED ADAPTED REGIME)"),
        (25.0, "STRATOSPHERE 25 KM (UNDEREXPANDED HIGH-ALTITUDE FAN EXPANSION)"),
    ]

    for alt_km, title in stations:
        pa = get_ambient_pressure_at_altitude(alt_km)
        plume = plume_engine.simulate_plume(
            x_exit=contour.length,
            exit_radius=contour.exit_radius,
            exit_mach=state.exit_mach,
            p_exit=state.exit_pressure,
            chamber_pressure=state.chamber_pressure,
            p_ambient=pa,
            plume_length=8.0,
            num_cells=4,
        )

        thermal = cooling.solve_station_thermal_equilibrium(
            x=0.0,
            radius=throat_radius_m,
            mach=1.0,
            throat_radius=throat_radius_m,
            coolant_temp=120.0,
        )

        canvas = BrailleCanvas(
            char_width=76,
            char_height=22,
            x_min=-0.3,
            x_max=contour.length + 8.5,
            y_min=-contour.exit_radius * 2.6,
            y_max=contour.exit_radius * 2.6,
        )
        viz = RocketVisualizer(canvas)
        viz.draw_nozzle_contour(contour)
        viz.draw_exhaust_plume(plume)

        print("\n" + "-" * 76)
        print(f" >>> FLIGHT STATION: ALTITUDE {alt_km:4.1f} km (P_amb = {pa / 1e3:6.2f} kPa)")
        print(f"     {title}")
        print("-" * 76)
        print(canvas.render_to_string())
        print(viz.format_telemetry_hud(state, plume, thermal))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThermoProp: Supersonic De Laval Nozzle & Rocket Propulsion Workbench"
    )
    parser.add_argument(
        "--engine",
        choices=["hydrolox", "methalox", "kerolox", "hypergolic"],
        default="methalox",
        help="Propellant chemistry combination (default: methalox)",
    )
    parser.add_argument(
        "--pc",
        type=float,
        default=100.0,
        help="Combustion chamber pressure in bar (default: 100.0)",
    )
    parser.add_argument(
        "--throat-radius",
        type=float,
        default=0.10,
        help="Nozzle throat radius in meters (default: 0.10)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=40.0,
        help="Nozzle expansion ratio Area_exit / Area_throat (default: 40.0)",
    )

    args = parser.parse_args()
    run_ascent_mission_demo(
        propellant_name=args.engine,
        chamber_pressure_bar=args.pc,
        throat_radius_m=args.throat_radius,
        expansion_ratio=args.eps,
    )


if __name__ == "__main__":
    main()
