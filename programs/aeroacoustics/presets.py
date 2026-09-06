"""
AeroAcoustics Engineering Presets & Aircraft Flight Profiles
Curated aerodynamic acoustic scenarios spanning subsonic flight, sound barrier breakout,
and high-Mach supersonic sonic boom signatures.
Zero external dependencies, standard library only.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class AeroPreset:
    name: str
    description: str
    base_mach: float
    source_type: str        # 'monopole', 'dipole', 'quadrupole', 'n_wave'
    source_freq: float      # Hz
    source_amplitude: float # Pa
    trajectory_mode: str    # 'straight', 'accelerating', 'circle', 'slalom'
    temp_celsius: float     # deg C
    peak_overpressure: float # Pa
    pulse_duration: float   # seconds


PRESETS: Dict[str, AeroPreset] = {
    "concorde_cruise": AeroPreset(
        name="Concorde Mach 2.04 Transatlantic Cruise",
        description="Classic civil supersonic transport cruising at 60,000 ft with distinct N-wave double boom signature.",
        base_mach=2.04,
        source_type="n_wave",
        source_freq=180.0,
        source_amplitude=85.0,
        trajectory_mode="straight",
        temp_celsius=-50.0,  # Stratospheric ambient temperature
        peak_overpressure=105.0,
        pulse_duration=0.12
    ),

    "sr71_blackbird": AeroPreset(
        name="SR-71 Blackbird Mach 3.2 High-Altitude Dash",
        description="Hypersonic reconnaissance profile with razor-thin Mach cone angle of 18.2 degrees and severe shock overpressure.",
        base_mach=3.20,
        source_type="n_wave",
        source_freq=240.0,
        source_amplitude=120.0,
        trajectory_mode="straight",
        temp_celsius=-55.0,
        peak_overpressure=160.0,
        pulse_duration=0.09
    ),

    "barrier_breakout": AeroPreset(
        name="F-16 Falcon Transonic Sound Barrier Breakout",
        description="Continuous acceleration from subsonic Mach 0.6 through transonic Mach 1.0 to supersonic Mach 1.4.",
        base_mach=0.60,
        source_type="n_wave",
        source_freq=220.0,
        source_amplitude=70.0,
        trajectory_mode="accelerating",
        temp_celsius=15.0,
        peak_overpressure=90.0,
        pulse_duration=0.08
    ),

    "airliner_approach": AeroPreset(
        name="Commercial Jetliner Subsonic Approach (Mach 0.25)",
        description="Clean subsonic aerodynamic lift dipole and turbofan engine acoustic tone with classic Doppler pitch descent.",
        base_mach=0.25,
        source_type="dipole",
        source_freq=330.0,
        source_amplitude=45.0,
        trajectory_mode="straight",
        temp_celsius=20.0,
        peak_overpressure=25.0,
        pulse_duration=0.05
    ),

    "nasa_x59": AeroPreset(
        name="NASA X-59 QueSST Quiet Supersonic Demonstrator",
        description="Low-boom shaped acoustic profile designed to replace the abrupt N-wave shock with a gentle sonic thump.",
        base_mach=1.42,
        source_type="n_wave",
        source_freq=150.0,
        source_amplitude=35.0,
        trajectory_mode="straight",
        temp_celsius=-15.0,
        peak_overpressure=32.0,  # Greatly suppressed peak overpressure
        pulse_duration=0.14
    ),

    "aerobatic_vortex": AeroPreset(
        name="Aerobatic Jet Vortex Shedding Quadrupole",
        description="High-G circular maneuver producing acoustic quadrupole radiation from unsteady turbulent trailing wingtip vortices.",
        base_mach=0.65,
        source_type="quadrupole",
        source_freq=440.0,
        source_amplitude=60.0,
        trajectory_mode="circle",
        temp_celsius=22.0,
        peak_overpressure=40.0,
        pulse_duration=0.06
    ),

    "slalom_evasion": AeroPreset(
        name="Supersonic Slalom Evasive Flight Maneuver",
        description="Supersonic sinusoidal flight path producing curved, intersecting Mach shock waves and dynamic Doppler warping.",
        base_mach=1.65,
        source_type="monopole",
        source_freq=280.0,
        source_amplitude=80.0,
        trajectory_mode="slalom",
        temp_celsius=10.0,
        peak_overpressure=95.0,
        pulse_duration=0.07
    )
}


def get_preset_list() -> List[str]:
    """Returns list of preset dictionary keys."""
    return list(PRESETS.keys())


def load_preset(key: str) -> AeroPreset:
    """Retrieves a specific aerodynamic acoustic preset."""
    return PRESETS.get(key, PRESETS["concorde_cruise"])
