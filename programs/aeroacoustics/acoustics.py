"""
Computational Aeroacoustics & Wave Propagation Kernel
First-principles simulation of acoustic wave propagation, Doppler shifts,
supersonic Mach cones, sonic boom N-wave signatures, and moving multipole sources.
Zero external dependencies, standard library only.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any


# Standard atmospheric constants at sea level (15 deg C, 101.325 kPa)
DEFAULT_AIR_DENSITY = 1.225       # kg/m^3 (rho_0)
DEFAULT_SOUND_SPEED = 343.0       # m/s (c_0)
REFERENCE_PRESSURE = 20e-6        # 20 micro-Pascals (threshold of human hearing)


def sound_speed_from_temp(temp_celsius: float) -> float:
    """
    Computes speed of sound in dry air from ambient temperature in Celsius:
    c = sqrt(gamma * R * T_kelvin) where gamma=1.4, R=287.05 J/(kg*K).
    """
    kelvin = max(1.0, temp_celsius + 273.15)
    return math.sqrt(1.4 * 287.05 * kelvin)


def sound_pressure_level_db(p_rms: float, p_ref: float = REFERENCE_PRESSURE) -> float:
    """
    Calculates Sound Pressure Level (SPL) in decibels relative to reference pressure:
    SPL = 20 * log10(max(p_rms, 1e-12) / p_ref).
    """
    val = max(1e-12, abs(p_rms))
    return 20.0 * math.log10(val / p_ref)


def mach_angle_rad(mach: float) -> Optional[float]:
    """
    Computes the Mach cone half-angle mu in radians:
    sin(mu) = 1 / M, defined only for supersonic flow (M >= 1.0).
    Returns None if subsonic (M < 1.0).
    """
    if mach < 1.0:
        return None
    sin_mu = min(1.0, 1.0 / max(1.0, mach))
    return math.asin(sin_mu)


def doppler_observed_frequency(f_source: float, mach: float, angle_rad: float) -> float:
    """
    Classical moving-source Doppler frequency shift:
    f_obs = f_0 / (1 - M * cos(theta))
    For supersonic M*cos(theta) >= 1, returns compressed/infinite frequency representation.
    """
    denom = 1.0 - mach * math.cos(angle_rad)
    if abs(denom) < 1e-4:
        return f_source * 1000.0
    return f_source / abs(denom)


@dataclass
class WavePacket:
    """
    Discrete expanding spherical acoustic wave envelope emitted at time t_emit.
    """
    x_emit: float
    y_emit: float
    t_emit: float
    frequency: float
    amplitude: float
    phase_offset: float = 0.0
    source_mach: float = 0.0
    is_shock: bool = False

    def radius_at(self, t_curr: float, c_sound: float) -> float:
        """Computes current spherical wavefront radius r(t) = c * (t - t_emit)."""
        dt = t_curr - this_t if (this_t := self.t_emit) <= t_curr else 0.0
        return c_sound * dt

    def amplitude_at(self, t_curr: float, c_sound: float, attenuation_alpha: float = 0.0005) -> float:
        """
        Calculates geometric spherical 1/r spreading and atmospheric attenuation:
        A(r) = (A_0 / max(1, r)) * exp(-alpha * r).
        """
        r = self.radius_at(t_curr, c_sound)
        if r <= 0.0:
            return 0.0
        return (self.amplitude / max(1.0, r)) * math.exp(-attenuation_alpha * r)


@dataclass
class MicrophoneStation:
    """
    Stationary virtual acoustic sensor recording acoustic pressure history.
    """
    station_id: str
    x: float
    y: float
    max_history_len: int = 500
    pressure_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)
    spl_dBA: float = 0.0
    peak_overpressure_pa: float = 0.0

    def record_sample(self, t: float, pressure_pa: float):
        self.time_history.append(t)
        self.pressure_history.append(pressure_pa)
        if len(self.pressure_history) > self.max_history_len:
            self.pressure_history.pop(0)
            self.time_history.pop(0)

        # Update peak overpressure and RMS SPL
        abs_p = abs(pressure_pa)
        if abs_p > self.peak_overpressure_pa:
            self.peak_overpressure_pa = abs_p

        # Running RMS over last 50 samples
        recent = self.pressure_history[-50:]
        if recent:
            rms = math.sqrt(sum(p * p for p in recent) / len(recent))
            self.spl_dBA = sound_pressure_level_db(rms)

    def reset(self):
        self.pressure_history.clear()
        self.time_history.clear()
        self.spl_dBA = 0.0
        self.peak_overpressure_pa = 0.0


class FlightTrajectory:
    """
    Analytical and parametric flight path kinematics yielding position and velocity.
    """
    def __init__(self, mode: str = "straight"):
        self.mode = mode  # straight, accelerating, circle, slalom, supersonic_dash

    def evaluate(self, t: float, base_mach: float, c_sound: float) -> Tuple[float, float, float, float, float]:
        """
        Returns (x, y, vx, vy, current_mach) at time t.
        Coordinate system origin (0, 0) centered in spatial domain.
        """
        speed = base_mach * c_sound

        if self.mode == "straight":
            # Level horizontal flight left to right
            x = -400.0 + speed * t
            y = 0.0
            vx = speed
            vy = 0.0
            curr_mach = base_mach

        elif self.mode == "accelerating":
            # Smooth acceleration passing through the sound barrier Mach 1.0
            # M(t) = 0.5 + 1.2 * (t / 4.0)
            curr_mach = 0.5 + 0.35 * t
            curr_speed = curr_mach * c_sound
            x = -450.0 + 0.5 * (curr_speed + 0.5 * c_sound) * t
            y = 0.0
            vx = curr_speed
            vy = 0.0

        elif self.mode == "circle":
            # Circular holding pattern / orbit
            radius = 180.0
            omega = speed / max(1.0, radius)
            theta = omega * t
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vx = -radius * omega * math.sin(theta)
            vy = radius * omega * math.cos(theta)
            curr_mach = base_mach

        elif self.mode == "slalom":
            # Sinusoidal evasive maneuvers starting at zero cross-track deflection
            x = -400.0 + speed * t
            dx = x + 400.0
            wavelength = 280.0
            amplitude = 90.0
            k = 2.0 * math.pi / wavelength
            y = amplitude * math.sin(k * dx)
            vx = speed
            vy = amplitude * k * vx * math.cos(k * dx)
            curr_speed = math.hypot(vx, vy)
            curr_mach = curr_speed / c_sound

        else:
            # Default fallback
            x = -400.0 + speed * t
            y = 0.0
            vx = speed
            vy = 0.0
            curr_mach = base_mach

        return x, y, vx, vy, curr_mach


class AeroAcousticEngine:
    """
    Core computational simulation engine modeling:
    - Kinematic moving sound sources (monopole, dipole, quadrupole, sonic boom N-wave).
    - Discrete wavefront emission and spherical expansion.
    - Supersonic shock envelope and Mach angle calculations.
    - Continuous retarded-time acoustic pressure field calculation.
    - Virtual microphone stations and real-time audio sampling.
    """

    def __init__(self,
                 base_mach: float = 1.4,
                 source_freq: float = 220.0,
                 source_amplitude: float = 50.0,
                 temp_celsius: float = 15.0):
        self.base_mach = base_mach
        self.source_freq = source_freq
        self.source_amplitude = source_amplitude
        self.temp_celsius = temp_celsius
        self.c_sound = sound_speed_from_temp(temp_celsius)
        self.rho_0 = DEFAULT_AIR_DENSITY

        # Source acoustic radiation type:
        # 'monopole' (mass injection), 'dipole' (lift/drag force),
        # 'quadrupole' (turbulent shear), 'n_wave' (sonic boom overpressure)
        self.source_type = "monopole"

        # Trajectory controller
        self.trajectory = FlightTrajectory("straight")
        self.sim_time = 0.0
        self.emission_interval = 0.035  # emit wavefront every 35 ms
        self.last_emission_time = -1.0

        # Current kinematic state
        self.source_x = -400.0
        self.source_y = 0.0
        self.source_vx = base_mach * self.c_sound
        self.source_vy = 0.0
        self.current_mach = base_mach

        # Active wavefronts ring buffer
        self.wavefronts: List[WavePacket] = []
        self.max_wavefronts = 150

        # Virtual microphone stations
        self.microphones: List[MicrophoneStation] = [
            MicrophoneStation("MIC_01_GROUND", x=0.0, y=-160.0),
            MicrophoneStation("MIC_02_FLYBY", x=80.0, y=-60.0),
            MicrophoneStation("MIC_03_OVERHEAD", x=0.0, y=140.0)
        ]

        # N-wave sonic boom parameters (Whitham theory)
        self.n_wave_peak_overpressure = 120.0  # Pa (approx 135 dB peak)
        self.n_wave_duration = 0.08            # seconds (80 ms total period)
        self.n_wave_rise_time = 0.003          # seconds (3 ms shock front rise)

        # Sonic boom ground footprint trail
        self.ground_boom_events: List[Dict[str, float]] = []

    def reset(self):
        """Resets simulation time, wavefronts, and microphone buffers."""
        self.sim_time = 0.0
        self.last_emission_time = -1.0
        self.wavefronts.clear()
        self.ground_boom_events.clear()
        for mic in self.microphones:
            mic.reset()
        self.update_kinematics(0.0)

    def set_mach(self, mach: float):
        """Updates base Mach number and recalculates kinematics."""
        self.base_mach = max(0.05, min(5.0, mach))
        self.update_kinematics(self.sim_time)

    def set_temperature(self, temp_c: float):
        """Updates ambient temperature and sound speed."""
        self.temp_celsius = temp_c
        self.c_sound = sound_speed_from_temp(temp_c)
        self.update_kinematics(self.sim_time)

    def update_kinematics(self, t: float):
        """Evaluates trajectory at time t."""
        self.source_x, self.source_y, self.source_vx, self.source_vy, self.current_mach = (
            self.trajectory.evaluate(t, self.base_mach, self.c_sound)
        )

    def compute_n_wave_pressure(self, dt_shock: float) -> float:
        """
        Whitham N-Wave Sonic Boom signature profile over time relative to shock arrival:
        Returns pressure deviation delta P in Pascals.
        Signature: sharp rise to +P_peak, linear decrease to -P_peak over duration T,
        and rapid recompression shock back to ambient.
        """
        T = self.n_wave_duration
        tau = self.n_wave_rise_time
        P = self.n_wave_peak_overpressure

        if dt_shock < 0.0:
            return 0.0
        elif dt_shock < tau:
            # Bow shock compression
            return P * (dt_shock / tau)
        elif dt_shock < T - tau:
            # Linear expansion between shocks
            rel_t = (dt_shock - tau) / (T - 2.0 * tau)
            return P * (1.0 - 2.0 * rel_t)
        elif dt_shock < T:
            # Tail shock recompression
            rel_t = (dt_shock - (T - tau)) / tau
            return -P * (1.0 - rel_t)
        else:
            return 0.0

    def compute_field_pressure_at(self, x: float, y: float, t: float) -> float:
        """
        Evaluates acoustic pressure p(x, y, t) via retarded-time summation of wavefronts.
        Accounts for source directivity (monopole vs dipole vs quadrupole).
        """
        total_p = 0.0

        for wf in self.wavefronts:
            dx = x - wf.x_emit
            dy = y - wf.y_emit
            dist = math.hypot(dx, dy)
            wf_radius = wf.radius_at(t, self.c_sound)
            diff_r = dist - wf_radius

            # Spatial thickness envelope of emitted wave packet
            pulse_width = 18.0
            if abs(diff_r) < pulse_width:
                spatial_env = math.cos(0.5 * math.pi * (diff_r / pulse_width)) ** 2
                time_elapsed = t - wf.t_emit

                # Phase and carrier oscillation
                phase = 2.0 * math.pi * wf.frequency * time_elapsed + wf.phase_offset

                # Base acoustic amplitude
                amp = wf.amplitude_at(t, self.c_sound)

                # Directional directivity factor D(theta)
                theta = math.atan2(dy, dx)
                source_angle = math.atan2(self.source_vy, self.source_vx) if (self.source_vx != 0 or self.source_vy != 0) else 0.0
                rel_theta = theta - source_angle

                if self.source_type == "monopole":
                    directivity = 1.0  # Omnidirectional
                elif self.source_type == "dipole":
                    directivity = math.sin(rel_theta)  # Figure-eight lift dipole
                elif self.source_type == "quadrupole":
                    directivity = 0.5 * math.sin(2.0 * rel_theta)  # Four-leaf quadrupole
                elif self.source_type == "n_wave":
                    # For N-wave, shock is concentrated near Mach cone
                    directivity = 1.2 if wf.source_mach >= 1.0 else 0.5
                else:
                    directivity = 1.0

                p_sample = amp * spatial_env * math.sin(phase) * directivity
                total_p += p_sample

        # In N-wave mode, add direct Whitham bow/tail shock overpressure when near shock front
        if self.source_type == "n_wave" and self.current_mach >= 1.0:
            # Distance from point (x, y) to current Mach line envelope
            mu = mach_angle_rad(self.current_mach)
            if mu is not None:
                dx_s = x - self.source_x
                dy_s = abs(y - self.source_y)
                # Mach cone boundary: dx_s = -dy_s / tan(mu)
                cone_x = -dy_s / math.tan(mu)
                dist_to_cone = dx_s - cone_x
                if -25.0 < dist_to_cone < 5.0:
                    dt_rel = (5.0 - dist_to_cone) / (self.c_sound * 1.5)
                    total_p += self.compute_n_wave_pressure(dt_rel)

        return total_p

    def step(self, dt: float):
        """
        Advances the simulation forward by time step dt:
        - Updates kinematics.
        - Emits new acoustic wavefronts.
        - Prunes distant decayed wavefronts.
        - Samples pressure at all virtual microphone sensors.
        """
        self.sim_time += dt
        self.update_kinematics(self.sim_time)

        # Check for periodic wavefront emission
        if self.sim_time - self.last_emission_time >= self.emission_interval:
            is_shock = self.current_mach >= 1.0
            wf = WavePacket(
                x_emit=self.source_x,
                y_emit=self.source_y,
                t_emit=self.sim_time,
                frequency=self.source_freq,
                amplitude=self.source_amplitude,
                source_mach=self.current_mach,
                is_shock=is_shock
            )
            self.wavefronts.append(wf)
            self.last_emission_time = self.sim_time

            # Record ground boom event along flight path if supersonic
            if is_shock:
                self.ground_boom_events.append({
                    "x": self.source_x,
                    "t": self.sim_time,
                    "mach": self.current_mach
                })
                if len(self.ground_boom_events) > 80:
                    self.ground_boom_events.pop(0)

        # Prune old wavefronts that have propagated outside simulation bounds
        max_dist = 900.0
        self.wavefronts = [
            wf for wf in self.wavefronts
            if wf.radius_at(self.sim_time, self.c_sound) < max_dist
        ]
        if len(self.wavefronts) > self.max_wavefronts:
            self.wavefronts = self.wavefronts[-self.max_wavefronts:]

        # Sample microphone sensors
        for mic in self.microphones:
            p = self.compute_field_pressure_at(mic.x, mic.y, self.sim_time)
            mic.record_sample(self.sim_time, p)

    def get_mach_cone_geometry(self) -> Optional[Dict[str, Any]]:
        """
        Returns geometric line coordinates for the supersonic Mach cone shock envelope.
        Returns None if flight is subsonic (M < 1.0).
        """
        if self.current_mach < 1.0:
            return None

        mu = mach_angle_rad(self.current_mach)
        if mu is None:
            return None

        # Direction of flight heading
        heading = math.atan2(self.source_vy, self.source_vx) if (self.source_vx != 0 or self.source_vy != 0) else 0.0

        # Rearward angle from heading
        rear_angle = heading + math.pi

        # Upper and lower Mach lines sweep backwards by (pi - mu)
        angle_upper = rear_angle + mu
        angle_lower = rear_angle - mu

        ray_len = 650.0
        x_up = self.source_x + ray_len * math.cos(angle_upper)
        y_up = self.source_y + ray_len * math.sin(angle_upper)
        x_low = self.source_x + ray_len * math.cos(angle_lower)
        y_low = self.source_y + ray_len * math.sin(angle_lower)

        return {
            "mu_rad": mu,
            "mu_deg": math.degrees(mu),
            "apex": (self.source_x, self.source_y),
            "line_upper": ((self.source_x, self.source_y), (x_up, y_up)),
            "line_lower": ((self.source_x, self.source_y), (x_low, y_low))
        }

    def compute_directivity_polar_pattern(self, num_points: int = 72) -> List[Tuple[float, float]]:
        """
        Computes the directional sound pressure pattern D(theta) over 360 degrees:
        Returns list of (angle_rad, normalized_intensity).
        """
        pattern = []
        heading = math.atan2(self.source_vy, self.source_vx) if (self.source_vx != 0 or self.source_vy != 0) else 0.0

        for i in range(num_points):
            theta = (i / num_points) * 2.0 * math.pi
            rel_angle = theta - heading

            # Convective amplification factor (1 - M * cos(rel_angle))^-4
            doppler_factor = 1.0 / max(0.12, abs(1.0 - self.current_mach * math.cos(rel_angle))) ** 2

            if self.source_type == "monopole":
                base = 1.0
            elif self.source_type == "dipole":
                base = abs(math.sin(rel_angle))
            elif self.source_type == "quadrupole":
                base = abs(math.sin(2.0 * rel_angle))
            elif self.source_type == "n_wave":
                base = 1.0 if self.current_mach < 1.0 else 1.5
            else:
                base = 1.0

            intensity = base * min(8.0, doppler_factor)
            pattern.append((theta, intensity))

        # Normalize pattern to peak 1.0
        max_int = max((p[1] for p in pattern), default=1.0)
        norm_factor = 1.0 / max(1e-5, max_int)
        return [(p[0], p[1] * norm_factor) for p in pattern]

    def compute_fft_spectrum(self, mic_index: int = 0) -> Tuple[List[float], List[float]]:
        """
        Computes discrete Fourier transform power spectrum of microphone pressure buffer.
        Returns (frequencies_hz, power_db). Standard library DFT implementation.
        """
        if mic_index >= len(self.microphones):
            return [], []

        mic = self.microphones[mic_index]
        history = mic.pressure_history
        N = len(history)
        if N < 32:
            return [], []

        # Zero-pad or truncate to power of 2 (up to 128 for real-time speed)
        M = min(128, 1 << (N.bit_length() - 1))
        samples = history[-M:]

        # Hanning window to reduce spectral leakage
        windowed = [
            samples[n] * 0.5 * (1.0 - math.cos(2.0 * math.pi * n / (M - 1)))
            for n in range(M)
        ]

        # Discrete Fourier Transform
        dt_avg = (mic.time_history[-1] - mic.time_history[-M]) / max(1, M - 1) if len(mic.time_history) >= M else 0.005
        sample_rate = 1.0 / max(1e-5, dt_avg)

        freqs = []
        powers = []
        half_M = M // 2

        for k in range(half_M):
            f_k = k * sample_rate / M
            re = sum(windowed[n] * math.cos(2.0 * math.pi * k * n / M) for n in range(M))
            im = sum(-windowed[n] * math.sin(2.0 * math.pi * k * n / M) for n in range(M))
            mag = math.sqrt(re * re + im * im) / M
            p_db = sound_pressure_level_db(mag)
            freqs.append(f_k)
            powers.append(p_db)

        return freqs, powers
