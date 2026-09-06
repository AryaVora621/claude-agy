#!/usr/bin/env python3
"""
OrbitMech: Mission Control & Astrodynamics Trajectory Optimization Laboratory.
Interactive demonstration showcasing:
  1. Spacecraft orbital state and Keplerian elements HUD.
  2. LEO-to-GEO Hohmann transfer with 3D Unicode Braille orbit rendering.
  3. Earth J2 oblateness secular nodal precession simulation.
  4. Jupiter gravity-assist hyperbolic flyby velocity boost.
  5. Earth-to-Mars interplanetary launch window optimization (Porkchop plot).
"""

import sys
import os
import time
import math

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
    SUN,
    JUPITER,
    MARS,
    AU_KM,
)
from orbitmech.kepler import (
    state_to_orbital_elements,
    orbital_elements_to_state,
    propagate_keplerian,
)
from orbitmech.propagator import (
    PerturbationConfig,
    propagate_symplectic_verlet,
    propagate_rk45,
)
from orbitmech.maneuvers import (
    hohmann_transfer,
    plane_change_maneuver,
    gravity_assist_flyby,
)
from orbitmech.porkchop import (
    generate_porkchop_grid,
)
from orbitmech.visualizer import (
    ANSI,
    Camera3D,
    render_orbit_scene,
    render_porkchop_contour,
    render_mission_telemetry_hud,
)


def run_mission_control():
    print("\n" + "=" * 76)
    print("        ORBITMECH: ASTRODYNAMICS & TRAJECTORY OPTIMIZATION LAB")
    print("   First-Principles Orbital Mechanics, Lambert Targeting & Porkchop Plots")
    print("=" * 76)

    # 1. Mission Telemetry HUD
    print("\n[1] Initial Orbital State & Flight Telemetry HUD:")
    leo_elements = ClassicalOrbitalElements(
        a=6778.137,          # 400 km altitude LEO
        e=0.012,             # Slight eccentricity
        i=math.radians(51.64), # ISS inclination
        raan=math.radians(110.0),
        arg_peri=math.radians(45.0),
        true_anomaly=math.radians(30.0),
        mu=EARTH.mu,
    )
    leo_state = orbital_elements_to_state(leo_elements)
    print(render_mission_telemetry_hud(leo_state, leo_elements, central_body=EARTH))

    # 2. LEO-to-GEO Hohmann Transfer & 3D Braille Orbit Rendering
    print("\n[2] LEO-to-GEO Coplanar Hohmann Transfer & 3D Orbital Trajectory:")
    r_leo = leo_elements.a
    r_geo = 42164.14
    hohmann_res = hohmann_transfer(r_leo, r_geo, mu=EARTH.mu)
    print(f" -> Departure Burn (Δv1):  {hohmann_res.delta_v1:.3f} km/s (at {r_leo:,.1f} km)")
    print(f" -> Arrival Burn (Δv2):    {hohmann_res.delta_v2:.3f} km/s (at {r_geo:,.1f} km)")
    print(f" -> Total Transfer ΔV:     {hohmann_res.total_delta_v:.3f} km/s")
    print(f" -> Time of Flight:        {hohmann_res.time_of_flight / 3600.0:.2f} hours")
    print(f" -> Required Phase Angle:  {hohmann_res.phase_angle_deg:.2f}°")

    # Generate Hohmann transfer arc trajectory for 3D visualization
    tx_elements = ClassicalOrbitalElements(
        a=hohmann_res.transfer_semi_major_axis,
        e=(r_geo - r_leo) / (r_geo + r_leo),
        i=math.radians(20.0),
        raan=math.radians(35.0),
        arg_peri=0.0,
        true_anomaly=0.0,
        mu=EARTH.mu,
    )
    tx_state = orbital_elements_to_state(tx_elements)
    transfer_states = [
        propagate_keplerian(tx_state, dt=i * (hohmann_res.time_of_flight / 40.0), mu=EARTH.mu)
        for i in range(41)
    ]
    cam = Camera3D(azimuth_deg=45.0, elevation_deg=30.0, scale=1200.0)
    print("\n" + render_orbit_scene(transfer_states, central_body=EARTH, camera=cam, canvas_chars_x=64, canvas_chars_y=16))

    # 3. Earth J2 Zonal Oblateness Precession Simulation
    print("\n[3] Earth J2 Oblateness Secular Nodal Regression Simulation:")
    t0 = time.perf_counter()
    prop_duration = 12.0 * 3600.0  # 12 hours
    config_j2 = PerturbationConfig(enable_j2=True, enable_drag=False)
    j2_res = propagate_rk45(
        initial_state=leo_state,
        duration=prop_duration,
        initial_dt=30.0,
        tolerance=1e-8,
        config=config_j2,
    )
    elapsed = time.perf_counter() - t0
    final_coe = state_to_orbital_elements(j2_res.final_state, mu=EARTH.mu)

    # Theoretical secular nodal precession rate
    p = leo_elements.semi_latus_rectum
    n = leo_elements.mean_motion
    d_omega_dt = -1.5 * EARTH.j2 * ((EARTH.radius / p) ** 2) * n * math.cos(leo_elements.i)
    expected_d_raan_deg = math.degrees(d_omega_dt * prop_duration)
    measured_d_raan_deg = math.degrees(final_coe.raan - leo_elements.raan)

    print(f" -> Integrated {j2_res.step_count:,} RK45 adaptive steps across {prop_duration / 3600.0:.1f} hours in {elapsed*1000:.2f} ms")
    print(f" -> Theoretical Nodal Drift: {expected_d_raan_deg:.4f}°")
    print(f" -> Measured Numerical Drift: {measured_d_raan_deg:.4f}° (Δ = {abs(measured_d_raan_deg - expected_d_raan_deg):.4f}°)")

    # 4. Planetary Gravity Assist Flyby (Jupiter)
    print("\n[4] Hyperbolic Planetary Gravity Assist (Jupiter Flyby):")
    v_jupiter = Vector3(0.0, 13.07, 0.0)
    v_sc_in = Vector3(9.5, 13.07, 0.0)
    flyby_res = gravity_assist_flyby(
        v_sc_in=v_sc_in,
        v_planet=v_jupiter,
        planet=JUPITER,
        periapsis_altitude=400000.0,  # 400,000 km altitude
    )
    print(f" -> Incoming Excess Speed (V_inf): {flyby_res.v_inf_in.norm():.3f} km/s")
    print(f" -> Hyperbolic Bending Angle (δ):   {flyby_res.bending_angle_deg:.2f}°")
    print(f" -> Closest Approach Altitude:      {flyby_res.periapsis_altitude:,.0f} km")
    print(f" -> Heliocentric ΔV Boost:          {flyby_res.delta_v_mag:.3f} km/s ({flyby_res.delta_v_mag * 3600:,.0f} km/h)")
    print(f" -> Heliocentric Outgoing Velocity: {flyby_res.v_sc_out.norm():.3f} km/s (Incoming: {v_sc_in.norm():.3f} km/s)")

    # 5. Earth-to-Mars Interplanetary Porkchop Plot
    print("\n[5] Earth-to-Mars Interplanetary Porkchop Plot Optimization:")
    t0 = time.perf_counter()
    porkchop_grid = generate_porkchop_grid(
        dep_start_day=0.0,
        dep_end_day=60.0,
        dep_steps=16,
        arr_start_day=160.0,
        arr_end_day=320.0,
        arr_steps=16,
    )
    elapsed_porkchop = time.perf_counter() - t0
    print(f" -> Evaluated {len(porkchop_grid.departure_days)*len(porkchop_grid.arrival_days)} Lambert solutions in {elapsed_porkchop*1000:.2f} ms")
    print(porkchop_grid.summary())

    # Render terminal Porkchop contour map
    dv_matrix = porkchop_grid.get_contour_matrix("total_delta_v")
    # Find min point index
    min_i, min_j = 0, 0
    min_val = float("inf")
    for i in range(len(dv_matrix)):
        for j in range(len(dv_matrix[0])):
            if dv_matrix[i][j] < min_val:
                min_val = dv_matrix[i][j]
                min_i, min_j = i, j

    contour_map = render_porkchop_contour(
        grid_matrix=dv_matrix,
        dep_days=porkchop_grid.departure_days,
        arr_days=porkchop_grid.arrival_days,
        min_point_idx=(min_i, min_j),
        title="EARTH -> MARS INTERPLANETARY PORKCHOP PLOT (TOTAL ΔV km/s)",
    )
    print(contour_map)

    print("=" * 76)
    print("            MISSION CONTROL SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_mission_control()
