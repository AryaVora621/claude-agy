"""
VeloSLAM: Analytical Dubins Path Planning Engine.
Solves minimum-length continuous curvature paths for non-holonomic vehicles
across the 6 canonical Dubins words: LSL, RSR, LSR, RSL, RLR, LRL.
"""

import math
from typing import List, Tuple, Optional
from veloslam.linalg import normalize_angle


def _mod2pi(theta: float) -> float:
    """Wraps angle to [0, 2*pi)."""
    return theta % (2.0 * math.pi)


class DubinsPath:
    """Represents an analytical Dubins trajectory."""

    def __init__(
        self,
        word: str,
        t: float,
        p: float,
        q: float,
        length: float,
        rho: float,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float]
    ) -> None:
        self.word = word        # e.g. "LSL", "RSR", "LSR", "RSL", "RLR", "LRL"
        self.t = t              # Length of segment 1 (radians or normalized length)
        self.p = p              # Length of segment 2
        self.q = q              # Length of segment 3
        self.length = length    # Total metric path length in meters
        self.rho = rho          # Turning radius in meters
        self.start_pose = start_pose
        self.goal_pose = goal_pose

    def sample_path(self, step_size: float = 0.1) -> List[Tuple[float, float, float]]:
        """
        Samples poses (x, y, theta) along the Dubins path at regular distance increments.
        """
        poses: List[Tuple[float, float, float]] = []
        num_steps = max(2, int(math.ceil(self.length / step_size)))

        for i in range(num_steps + 1):
            s = min(self.length, i * step_size)
            pose = self.evaluate(s)
            poses.append(pose)

        return poses

    def evaluate(self, s: float) -> Tuple[float, float, float]:
        """Evaluates vehicle pose (x, y, theta) at distance s along the path."""
        s = max(0.0, min(self.length, s))
        x0, y0, th0 = self.start_pose
        rho = self.rho

        seg_lengths = [self.t * rho, self.p * rho, self.q * rho]
        types = list(self.word)

        curr_x, curr_y, curr_th = x0, y0, th0
        dist_left = s

        for seg_len, seg_type in zip(seg_lengths, types):
            if dist_left <= 0:
                break
            step = min(dist_left, seg_len)

            if seg_type == "S":
                curr_x += step * math.cos(curr_th)
                curr_y += step * math.sin(curr_th)
            elif seg_type == "L":
                phi = step / rho
                curr_x += rho * (math.sin(curr_th + phi) - math.sin(curr_th))
                curr_y += rho * (-math.cos(curr_th + phi) + math.cos(curr_th))
                curr_th = normalize_angle(curr_th + phi)
            elif seg_type == "R":
                phi = step / rho
                curr_x += rho * (-math.sin(curr_th - phi) + math.sin(curr_th))
                curr_y += rho * (math.cos(curr_th - phi) - math.cos(curr_th))
                curr_th = normalize_angle(curr_th - phi)

            dist_left -= step

        return curr_x, curr_y, curr_th


def _dubins_LSL(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    tmp0 = d + sa - sb
    p_sq = 2.0 + d * d - 2.0 * (ca * cb + sa * sb) + 2.0 * d * (sa - sb)
    if p_sq < 0:
        return float("inf"), 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    tmp1 = math.atan2(cb - ca, tmp0)
    t = _mod2pi(-alpha + tmp1)
    q = _mod2pi(beta - tmp1)
    return t + p + q, t, p, q


def _dubins_RSR(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    tmp0 = d - sa + sb
    p_sq = 2.0 + d * d - 2.0 * (ca * cb + sa * sb) - 2.0 * d * (sa - sb)
    if p_sq < 0:
        return float("inf"), 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    tmp1 = math.atan2(ca - cb, tmp0)
    t = _mod2pi(alpha - tmp1)
    q = _mod2pi(-beta + tmp1)
    return t + p + q, t, p, q


def _dubins_LSR(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_sq = -2.0 + d * d + 2.0 * (ca * cb + sa * sb) + 2.0 * d * (sa + sb)
    if p_sq < 0:
        return float("inf"), 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    tmp1 = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    t = _mod2pi(-alpha + tmp1)
    q = _mod2pi(-beta + tmp1)
    return t + p + q, t, p, q


def _dubins_RSL(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_sq = -2.0 + d * d + 2.0 * (ca * cb + sa * sb) - 2.0 * d * (sa + sb)
    if p_sq < 0:
        return float("inf"), 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    tmp1 = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    t = _mod2pi(alpha - tmp1)
    q = _mod2pi(beta - tmp1)
    return t + p + q, t, p, q


def _dubins_RLR(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    tmp0 = (6.0 - d * d + 2.0 * (ca * cb + sa * sb) + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp0) > 1.0:
        return float("inf"), 0.0, 0.0, 0.0
    p = _mod2pi(2.0 * math.pi - math.acos(tmp0))
    t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p * 0.5)
    q = _mod2pi(alpha - beta - t + p)
    return t + p + q, t, p, q


def _dubins_LRL(alpha: float, beta: float, d: float) -> Tuple[float, float, float, float]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    tmp0 = (6.0 - d * d + 2.0 * (ca * cb + sa * sb) - 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp0) > 1.0:
        return float("inf"), 0.0, 0.0, 0.0
    p = _mod2pi(2.0 * math.pi - math.acos(tmp0))
    t = _mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + p * 0.5)
    q = _mod2pi(beta - alpha - t + p)
    return t + p + q, t, p, q


def dubins_shortest_path(
    start_pose: Tuple[float, float, float],
    goal_pose: Tuple[float, float, float],
    rho: float
) -> DubinsPath:
    """
    Computes the exact minimum-length Dubins path connecting start_pose to goal_pose
    with minimum curvature radius rho.
    start_pose: (x, y, theta)
    goal_pose: (x, y, theta)
    """
    x1, y1, th1 = start_pose
    x2, y2, th2 = goal_pose

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    d = dist / rho
    th = math.atan2(dy, dx)

    alpha = _mod2pi(th1 - th)
    beta = _mod2pi(th2 - th)

    solvers = [
        ("LSL", _dubins_LSL),
        ("RSR", _dubins_RSR),
        ("LSR", _dubins_LSR),
        ("RSL", _dubins_RSL),
        ("RLR", _dubins_RLR),
        ("LRL", _dubins_LRL),
    ]

    best_len = float("inf")
    best_word = "LSL"
    best_t, best_p, best_q = 0.0, 0.0, 0.0

    for word, solver in solvers:
        cost, t, p, q = solver(alpha, beta, d)
        if cost < best_len:
            best_len = cost
            best_word = word
            best_t, best_p, best_q = t, p, q

    total_length = best_len * rho
    return DubinsPath(
        word=best_word,
        t=best_t,
        p=best_p,
        q=best_q,
        length=total_length,
        rho=rho,
        start_pose=start_pose,
        goal_pose=goal_pose
    )
