"""Neuromorphic event-based vision and Dynamic Vision Sensor (DVS) stream processor.

Implements Address-Event Representation (AER), synthetic neuromorphic stimuli generators,
spatial-temporal refractory and background activity (BAF) filters,
the Surface of Active Events (SAE / Time Surface), and event-based optical flow estimation.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DVSEvent:
    """A single neuromorphic Address-Event Representation (AER) packet."""
    x: int              # Horizontal pixel coordinate [0..width-1]
    y: int              # Vertical pixel coordinate [0..height-1]
    timestamp_us: int   # Event timestamp in microseconds
    polarity: int       # Event polarity: +1 for ON (brightening), -1 for OFF (darkening)


class DVSStream:
    """A temporal sequence of neuromorphic DVS events."""

    def __init__(
        self,
        events: Optional[List[DVSEvent]] = None,
        width: int = 128,
        height: int = 128,
    ) -> None:
        self.events: List[DVSEvent] = events or []
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.events)

    def append(self, event: DVSEvent) -> None:
        self.events.append(event)

    def slice_time(self, start_us: int, end_us: int) -> DVSStream:
        """Return sub-stream of events within the timestamp window [start_us, end_us)."""
        filtered = [e for e in self.events if start_us <= e.timestamp_us < end_us]
        return DVSStream(filtered, self.width, self.height)

    def count_polarities(self) -> Tuple[int, int]:
        """Return (num_on_events, num_off_events)."""
        on_count = sum(1 for e in self.events if e.polarity > 0)
        off_count = sum(1 for e in self.events if e.polarity < 0)
        return on_count, off_count

    @property
    def duration_us(self) -> int:
        if not self.events:
            return 0
        return self.events[-1].timestamp_us - self.events[0].timestamp_us


class RefractoryFilter:
    """Clamps maximum firing rate per pixel by discarding events within tau_ref_us."""

    def __init__(self, width: int, height: int, tau_ref_us: int = 5000) -> None:
        self.width = width
        self.height = height
        self.tau_ref_us = tau_ref_us
        self.last_timestamps: List[List[int]] = [
            [-10000000 for _ in range(width)] for _ in range(height)
        ]

    def filter_event(self, event: DVSEvent) -> bool:
        """Returns True if the event passes the refractory filter, False if dropped."""
        if 0 <= event.x < self.width and 0 <= event.y < self.height:
            last_t = self.last_timestamps[event.y][event.x]
            if event.timestamp_us - last_t >= self.tau_ref_us:
                self.last_timestamps[event.y][event.x] = event.timestamp_us
                return True
            return False
        return False

    def process_stream(self, stream: DVSStream) -> DVSStream:
        passed = [e for e in stream.events if self.filter_event(e)]
        return DVSStream(passed, stream.width, stream.height)


class BackgroundActivityFilter:
    """Filters uncorrelated thermal/noise events lacking spatio-temporal neighbors."""

    def __init__(
        self, width: int, height: int, window_us: int = 10000, radius: int = 1
    ) -> None:
        self.width = width
        self.height = height
        self.window_us = window_us
        self.radius = radius
        self.last_timestamps: List[List[int]] = [
            [-10000000 for _ in range(width)] for _ in range(height)
        ]

    def filter_event(self, event: DVSEvent) -> bool:
        """Check if any neighboring pixel had an event within window_us."""
        x, y, t = event.x, event.y, event.timestamp_us
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        has_neighbor = False
        y_min = max(0, y - self.radius)
        y_max = min(self.height, y + self.radius + 1)
        x_min = max(0, x - self.radius)
        x_max = min(self.width, x + self.radius + 1)

        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if nx == x and ny == y:
                    continue
                if t - self.last_timestamps[ny][nx] <= self.window_us:
                    has_neighbor = True
                    break
            if has_neighbor:
                break

        # Always update current pixel's timestamp
        self.last_timestamps[y][x] = t
        return has_neighbor

    def process_stream(self, stream: DVSStream) -> DVSStream:
        passed = [e for e in stream.events if self.filter_event(e)]
        return DVSStream(passed, stream.width, stream.height)


class SurfaceOfActiveEvents:
    """Surface of Active Events (SAE) / Time Surface.

    Maintains the timestamp of the most recent event at each pixel location.
    Provides exponentially decayed time surfaces representing recent optical motion.
    """

    def __init__(self, width: int, height: int, tau_decay_us: float = 30000.0) -> None:
        self.width = width
        self.height = height
        self.tau_decay_us = tau_decay_us

        # Separate time surfaces for ON (+1) and OFF (-1) polarities
        self.sae_on: List[List[int]] = [[-10000000 for _ in range(width)] for _ in range(height)]
        self.sae_off: List[List[int]] = [[-10000000 for _ in range(width)] for _ in range(height)]
        self.latest_timestamp_us: int = 0

    def update(self, event: DVSEvent) -> None:
        """Update the SAE with an arriving event."""
        if 0 <= event.x < self.width and 0 <= event.y < self.height:
            if event.polarity > 0:
                self.sae_on[event.y][event.x] = event.timestamp_us
            else:
                self.sae_off[event.y][event.x] = event.timestamp_us

            if event.timestamp_us > self.latest_timestamp_us:
                self.latest_timestamp_us = event.timestamp_us

    def get_decayed_surface(
        self, current_time_us: Optional[int] = None, polarity: int = 1
    ) -> List[List[float]]:
        """Return 2D normalized exponential time surface in range [0.0, 1.0]."""
        t_ref = current_time_us if current_time_us is not None else self.latest_timestamp_us
        surface = self.sae_on if polarity > 0 else self.sae_off

        decayed: List[List[float]] = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                dt = t_ref - surface[y][x]
                if dt >= 0:
                    val = math.exp(-dt / self.tau_decay_us)
                    decayed[y][x] = max(0.0, min(1.0, val))
        return decayed


class OpticalFlowEstimator:
    """Event-based normal optical flow estimation via local plane fitting on the SAE."""

    def __init__(
        self, width: int, height: int, patch_radius: int = 2, min_events: int = 5
    ) -> None:
        self.width = width
        self.height = height
        self.patch_radius = patch_radius
        self.min_events = min_events
        self.sae = SurfaceOfActiveEvents(width, height)

    def estimate_flow(self, event: DVSEvent) -> Optional[Tuple[float, float]]:
        """Fit a local plane to recent timestamps around the event:

        Plane: a * (x - x0) + b * (y - y0) + c = t - t0
        Velocity: v = (a, b) / (a^2 + b^2) in pixels per second.
        Returns (vx, vy) or None if plane fit is degenerate.
        """
        self.sae.update(event)

        x0, y0, t0 = event.x, event.y, event.timestamp_us
        surface = self.sae.sae_on if event.polarity > 0 else self.sae.sae_off

        pts_x = []
        pts_y = []
        pts_t = []

        r = self.patch_radius
        for dy in range(-r, r + 1):
            ny = y0 + dy
            if 0 <= ny < self.height:
                for dx in range(-r, r + 1):
                    nx = x0 + dx
                    if 0 <= nx < self.width:
                        t_val = surface[ny][nx]
                        # Only consider events within 50ms of current event
                        if 0 <= t0 - t_val < 50000:
                            pts_x.append(dx)
                            pts_y.append(dy)
                            pts_t.append((t_val - t0) * 1e-6)  # Convert us to seconds

        if len(pts_x) < self.min_events:
            return None

        # Least-squares fit of a * dx + b * dy + c = dt
        # Normal equations: (A^T A) [a, b, c]^T = A^T dt
        n = len(pts_x)
        sum_x = sum(pts_x)
        sum_y = sum(pts_y)
        sum_xx = sum(x * x for x in pts_x)
        sum_yy = sum(y * y for y in pts_y)
        sum_xy = sum(x * y for x, y in zip(pts_x, pts_y))
        sum_t = sum(pts_t)
        sum_xt = sum(x * t for x, t in zip(pts_x, pts_t))
        sum_yt = sum(y * t for y, t in zip(pts_y, pts_t))

        # 2x2 system centered about mean
        mean_x = sum_x / n
        mean_y = sum_y / n
        mean_t = sum_t / n

        s_xx = sum_xx - n * mean_x * mean_x
        s_yy = sum_yy - n * mean_y * mean_y
        s_xy = sum_xy - n * mean_x * mean_y
        s_xt = sum_xt - n * mean_x * mean_t
        s_yt = sum_yt - n * mean_y * mean_t

        det = s_xx * s_yy - s_xy * s_xy
        if abs(det) < 1e-9:
            return None

        a = (s_yy * s_xt - s_xy * s_yt) / det
        b = (s_xx * s_yt - s_xy * s_xt) / det

        norm_sq = a * a + b * b
        if norm_sq < 1e-6:
            return None

        # Normal velocity v = (a, b) / (a^2 + b^2) in pixels/sec
        vx = a / norm_sq
        vy = b / norm_sq

        # Filter out extreme outliers
        if math.hypot(vx, vy) > 5000.0:
            return None

        return vx, vy


class DVSStimulusGenerator:
    """Synthetic neuromorphic DVS event stream generators for benchmarking."""

    @staticmethod
    def moving_vertical_bar(
        width: int = 64,
        height: int = 64,
        speed_px_s: float = 50.0,
        duration_s: float = 0.5,
        bar_thickness: int = 4,
    ) -> DVSStream:
        """Generate DVS events for a vertical bar moving horizontally from left to right."""
        stream = DVSStream(width=width, height=height)
        total_time_us = int(duration_s * 1e6)
        dt_us = int(1e6 / (speed_px_s * 4))  # Sub-pixel sampling rate

        cur_t_us = 0
        while cur_t_us < total_time_us:
            pos_x = (cur_t_us * 1e-6) * speed_px_s
            lead_x = int(pos_x)
            trail_x = lead_x - bar_thickness

            if 0 <= lead_x < width:
                # Leading edge causes ON events
                for y in range(0, height, 2):
                    stream.append(DVSEvent(lead_x, y, cur_t_us, +1))

            if 0 <= trail_x < width:
                # Trailing edge causes OFF events
                for y in range(0, height, 2):
                    stream.append(DVSEvent(trail_x, y, cur_t_us, -1))

            cur_t_us += dt_us

        return stream

    @staticmethod
    def rotating_line(
        width: int = 64,
        height: int = 64,
        rpm: float = 120.0,
        duration_s: float = 0.5,
    ) -> DVSStream:
        """Generate DVS events for a line rotating about the center."""
        stream = DVSStream(width=width, height=height)
        cx, cy = width / 2.0, height / 2.0
        radius = min(cx, cy) * 0.8
        omega = (rpm * 2.0 * math.pi) / 60.0  # rad/s

        total_time_us = int(duration_s * 1e6)
        dt_us = 2000  # 2ms steps

        cur_t_us = 0
        prev_theta = 0.0
        while cur_t_us < total_time_us:
            theta = (cur_t_us * 1e-6) * omega
            for r in range(4, int(radius), 2):
                x = int(cx + r * math.cos(theta))
                y = int(cy + r * math.sin(theta))
                if 0 <= x < width and 0 <= y < height:
                    stream.append(DVSEvent(x, y, cur_t_us, +1))
            cur_t_us += dt_us

        return stream

    @staticmethod
    def add_poisson_noise(
        stream: DVSStream, noise_rate_hz: float = 2.0
    ) -> DVSStream:
        """Inject uncorrelated background Poisson noise events."""
        noisy = DVSStream(list(stream.events), stream.width, stream.height)
        total_time_s = stream.duration_us * 1e-6
        num_noise_events = int(stream.width * stream.height * noise_rate_hz * total_time_s)

        for _ in range(num_noise_events):
            rx = random.randint(0, stream.width - 1)
            ry = random.randint(0, stream.height - 1)
            rt = random.randint(0, max(1, stream.duration_us))
            rp = 1 if random.random() > 0.5 else -1
            noisy.append(DVSEvent(rx, ry, rt, rp))

        # Re-sort chronologically by timestamp
        noisy.events.sort(key=lambda e: e.timestamp_us)
        return noisy
