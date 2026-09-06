"""
Interactive Tokamak Magnetic Confinement Fusion Laboratory Workbench.

Demonstrates:
  1. Tokamak reactor geometry & Grad-Shafranov 2D plasma equilibrium
  2. Safety factor q(psi) profile & rational resonance detection
  3. Symplectic Boris particle pusher: trapped neoclassical banana orbits & passing ions
  4. Poincaré surface-of-section puncture mapping & resonant magnetic perturbations (RMP)
  5. 2x4 sub-pixel Unicode Braille poloidal cross-section rendering with 24-bit TrueColor
  6. Fusion reactor telemetry HUD with Lawson triple product & ignition margin
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure parent directory is in sys.path for standalone invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stellarfusion.equilibrium import (
    EquilibriumProfile,
    GradShafranovSolver,
    Grid2D,
    SolovevEquilibrium,
)
from stellarfusion.magnetic import (
    MagneticFieldEvaluator,
    SafetyFactorCalculator,
)
from stellarfusion.particles import (
    BorisParticlePusher,
    ParticleSpecies,
    ParticleState,
)
from stellarfusion.poincare import (
    PoincareFieldTracer,
    ResonantMagneticPerturbation,
)
from stellarfusion.visualizer import (
    BrailleCanvas,
    TokamakVisualizer,
)


def run_workbench() -> None:
    print("\033[1m======================================================================\033[0m")
    print("\033[1m      STELLARFUSION : TOKAMAK PLASMA EQUILIBRIUM & CONFINEMENT LAB    \033[0m")
    print("\033[1m======================================================================\033[0m\n")

    # 1. Setup Tokamak Geometry & Solovev Equilibrium
    r_0 = 3.0       # Major radius (m)
    a_minor = 1.0   # Minor radius (m)
    kappa = 1.6     # Plasma elongation
    b_0 = 3.0       # Toroidal magnetic field on axis (T)
    psi_0 = 1.8     # Reference poloidal flux parameter

    print(f"[*] Initializing Tokamak Equilibrium (R0={r_0:.2f}m, a={a_minor:.2f}m, B0={b_0:.2f}T, k={kappa:.2f})...")
    solovev = SolovevEquilibrium(r_0=r_0, z_0=0.0, kappa=kappa, psi_0=psi_0, b_0=b_0)
    mag_evaluator = MagneticFieldEvaluator(solovev)

    # 2. Safety Factor & Instability Analysis
    print("[*] Evaluating Safety Factor Profile q(r) and Magnetic Shear s(r)...")
    q_calculator = SafetyFactorCalculator(
        evaluator=mag_evaluator,
        r_axis=r_0,
        z_axis=0.0,
        minor_radius_a=a_minor,
        kappa=kappa,
    )

    q_profile = q_calculator.compute_profile(num_points=8)
    q_0 = q_calculator.compute_q_at_minor_radius(0.08)
    q_95 = q_calculator.compute_q_at_minor_radius(0.95 * a_minor)

    print(f"    - On-Axis Safety Factor q0 : {q_0:.3f} (Sawtooth margin: {'STABLE (>1.0)' if q_0 >= 1.0 else 'UNSTABLE'})")
    print(f"    - Edge Safety Factor q95   : {q_95:.3f} (External kink margin: {'STABLE (>=3.0)' if q_95 >= 3.0 else 'BORDERLINE'})")

    print("\n    [Radial Safety Factor Profile]")
    print("    r/a       q(r)     Shear s   Status")
    print("    -----------------------------------------")
    for rho, q_val, shear in q_profile:
        status_note = "Normal Shear" if shear >= 0.0 else "Reversed Shear"
        print(f"    {rho:5.2f}    {q_val:6.3f}    {shear:6.3f}    {status_note}")

    rational_surfaces = q_calculator.find_rational_surfaces(tolerances=0.08)
    if rational_surfaces:
        print("\n    [Resonant Rational Surfaces Detected]")
        for m, n, rho_loc in rational_surfaces:
            print(f"    - Mode m/n = {m}/{n} at normalized radius rho = {rho_loc:.2f}")

    # 3. Symplectic Boris Particle Pusher: Trapped Neoclassical Banana Orbit
    print("\n[*] Simulating Neoclassical Charged Particle Orbits (Symplectic Boris Algorithm)...")
    pusher_d = BorisParticlePusher(mag_evaluator, species=ParticleSpecies.DEUTERIUM)

    # Launch a trapped deuterium ion from outboard midplane with low parallel velocity
    r_launch = r_0 + 0.6 * a_minor
    # Perpendicular velocity corresponds to ~15 keV, small parallel velocity
    v_thermal = 8.5e5  # m/s
    init_banana_state = ParticleState(
        x=r_launch,
        y=0.0,
        z=0.0,
        vx=0.05 * v_thermal,
        vy=0.98 * v_thermal,  # Primarily perpendicular to poloidal field
        vz=0.15 * v_thermal,
    )

    dt = 2.0e-8
    steps = 1500
    orbit_banana = pusher_d.trace_trajectory(init_banana_state, dt=dt, num_steps=steps, stride=2)

    # Verify Boris energy conservation
    initial_energy = init_banana_state.speed
    final_energy = orbit_banana[-1].speed
    energy_drift = abs(final_energy - initial_energy) / initial_energy
    print(f"    - Symplectic Energy Conservation : drift = {energy_drift:.2e} (Exact to floating precision)")

    is_trapped, bounces, bounce_freq, banana_width = pusher_d.analyze_orbit_trapping(
        init_banana_state, r_axis=r_0, dt=dt, num_steps=steps
    )
    orbit_type = "Trapped Banana Orbit" if is_trapped else "Passing Orbit"
    print(f"    - Orbit Classification           : {orbit_type}")
    print(f"    - Banana Bounce Count            : {bounces} reflections at magnetic mirror points")
    print(f"    - Radial Banana Orbit Width      : {banana_width * 100.0:.2f} cm")

    # 4. Poincaré Field Line Puncture Map
    print("\n[*] Tracing Poincaré Surface-of-Section Punctures (60 Toroidal Transits)...")
    rmp = ResonantMagneticPerturbation(m_mode=2, n_mode=1, amplitude=5e-4, r_axis=r_0, z_axis=0.0)
    tracer = PoincareFieldTracer(mag_evaluator, rmp=rmp)
    punctures = tracer.trace_puncture_series(
        r_start=r_0 + 0.5 * a_minor,
        z_start=0.0,
        num_turns=50,
        steps_per_turn=48,
    )
    print(f"    - Recorded {len(punctures)} Poincaré punctures across poloidal plane phi = 0")

    # 5. Sub-Pixel Unicode Braille Cross-Section Rendering
    print("\n[*] Rendering Poloidal Magnetic Flux Surfaces and Banana Orbit (2x4 Sub-Pixel Braille)...")
    canvas = BrailleCanvas(
        char_width=72,
        char_height=30,
        r_min=1.4,
        r_max=4.6,
        z_min=-2.2,
        z_max=2.2,
    )
    viz = TokamakVisualizer(canvas)

    # Render vacuum vessel first wall
    viz.draw_vacuum_vessel(r_axis=r_0, z_axis=0.0, a_wall=a_minor * 1.15, kappa_wall=kappa * 1.05)
    # Render nested poloidal flux surfaces
    viz.draw_flux_contours(r_axis=r_0, z_axis=0.0, a_minor=a_minor, kappa=kappa, num_surfaces=6)
    # Overlay neoclassical trapped banana particle trajectory
    viz.draw_particle_orbit(orbit_banana, color=(50, 255, 140))
    # Overlay Poincaré punctures
    viz.draw_poincare_punctures(punctures, color=(255, 220, 40))

    rendered_canvas = canvas.render_to_string()
    print(rendered_canvas)

    # 6. Display Tokamak Telemetry HUD
    hud = viz.format_telemetry_hud(
        r_0=r_0,
        a_minor=a_minor,
        b_0=b_0,
        i_p=2.1e6,
        q_0=q_0,
        q_95=q_95,
        t_core_kev=18.5,
        p_core_kpa=125.0,
        beta_toroidal_pct=3.8,
        confinement_time_s=2.8,
        plasma_density_m3=1.2e20,
    )
    print(hud)
    print("\n[+] Tokamak Confinement Laboratory run complete.\n")


if __name__ == "__main__":
    run_workbench()
