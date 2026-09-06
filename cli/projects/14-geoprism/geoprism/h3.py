"""
GeoPrism: H3 Hierarchical Hexagonal Spatial Grid Engine.
Implements the Discrete Global Grid System (DGGS) with 64-bit bit-packed cell IDs,
aperture-7 hierarchical resolutions (0-15), coordinate conversions, and k-ring traversal.
"""

import math
from typing import List, Tuple, Set, Dict, Optional

# H3 Index Bit Layout Constants
H3_CELL_MODE = 1
H3_NUM_BASE_CELLS = 122
H3_MAX_RESOLUTION = 15

# Directions in aperture 7 grid (0: center, 1..6: adjacent neighbors)
DIRECTION_CENTER = 0
DIRECTION_K_AXES = (1, 2, 3, 4, 5, 6)

# Hexagonal neighbor coordinate offsets (in axial coordinates q, r)
# (q, r) where q is horizontal and r is 60-degree diagonal
HEX_DIRECTIONS = [
    (1, 0),    # East
    (1, -1),   # North-East
    (0, -1),   # North-West
    (-1, 0),   # West
    (-1, 1),   # South-West
    (0, 1)     # South-East
]


def encode_h3(resolution: int, base_cell: int, digits: List[int]) -> int:
    """
    Packs resolution, base_cell, and up to 15 directional digits into a 64-bit integer.
    Bit Layout:
      - Bit 63: 0
      - Bits 62-59: Mode (4 bits: 1)
      - Bits 58-52: Reserved (7 bits: 0)
      - Bits 51-48: Resolution (4 bits: 0-15)
      - Bits 47-41: Base cell (7 bits: 0-121)
      - Bits 40-0:  15 digits (3 bits each: 0-6 for used, 7 for unused)
    """
    if not (0 <= resolution <= H3_MAX_RESOLUTION):
        raise ValueError(f"Resolution must be 0-{H3_MAX_RESOLUTION}")
    if not (0 <= base_cell < H3_NUM_BASE_CELLS):
        raise ValueError(f"Base cell must be 0-{H3_NUM_BASE_CELLS - 1}")

    h3 = 0
    # Mode: 1
    h3 |= (H3_CELL_MODE & 0x0F) << 59
    # Resolution (4 bits at 52-55)
    h3 |= (resolution & 0x0F) << 52
    # Base cell (7 bits at 45-51)
    h3 |= (base_cell & 0x7F) << 45

    # Digits for resolution 1..15 (45 bits at 0-44)
    for r in range(1, 16):
        shift = 45 - (r * 3)
        if r <= resolution:
            val = digits[r - 1] if (r - 1) < len(digits) else 0
        else:
            val = 7  # Unused digit
        h3 |= (val & 0x07) << shift

    return h3


def decode_h3(h3_index: int) -> Tuple[int, int, List[int]]:
    """
    Unpacks a 64-bit H3 integer into (resolution, base_cell, list_of_digits).
    """
    mode = (h3_index >> 59) & 0x0F
    if mode != H3_CELL_MODE:
        raise ValueError("Invalid H3 index: mode is not H3_CELL")

    res = (h3_index >> 52) & 0x0F
    base_cell = (h3_index >> 45) & 0x7F

    digits: List[int] = []
    for r in range(1, res + 1):
        shift = 45 - (r * 3)
        d = (h3_index >> shift) & 0x07
        digits.append(d)

    return res, base_cell, digits


def h3_to_string(h3_index: int) -> str:
    """Formats 64-bit integer H3 index as standard 15-character hexadecimal string."""
    return f"{h3_index:015x}"


def string_to_h3(h3_str: str) -> int:
    """Parses standard hexadecimal string into 64-bit integer H3 index."""
    return int(h3_str, 16)


def h3_get_resolution(h3_index: int) -> int:
    return (h3_index >> 52) & 0x0F


def h3_get_base_cell(h3_index: int) -> int:
    return (h3_index >> 45) & 0x7F


# Earth geometry constants
EARTH_RADIUS_KM = 6371.0088


def _base_cell_for_lat_lon(lat: float, lon: float) -> Tuple[int, float, float]:
    """
    Maps geographic (lat, lon) to base cell ID (0-121) and localized planar coordinates (x, y).
    Partitions the sphere into 122 discrete spatial sectors.
    """
    # Normalize lon to [-180, 180], lat to [-90, 90]
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # 122 base cell distribution:
    # 10 latitude bands, roughly 12 cells per band
    band = min(9, max(0, int((lat + 90.0) / 18.0)))
    col = min(11, max(0, int((lon + 180.0) / 30.0)))
    base_cell = band * 12 + col

    # Local coordinates normalized to [-1.0, 1.0] within the base cell sector
    band_center_lat = -90.0 + (band + 0.5) * 18.0
    col_center_lon = -180.0 + (col + 0.5) * 30.0

    d_lat = (lat - band_center_lat) / 9.0   # in [-1, 1]
    d_lon = (lon - col_center_lon) / 15.0  # in [-1, 1]

    # Convert to hexagonal planar (x, y)
    x = d_lon * math.cos(lat_rad)
    y = d_lat
    return base_cell, x, y


def _lat_lon_from_base_cell(base_cell: int, x: float, y: float) -> Tuple[float, float]:
    """Reconstructs geographic (lat, lon) from base cell and local planar offset."""
    band = base_cell // 12
    col = base_cell % 12

    band_center_lat = -90.0 + (band + 0.5) * 18.0
    col_center_lon = -180.0 + (col + 0.5) * 30.0

    lat = band_center_lat + (y * 9.0)
    lat_rad = math.radians(lat)
    cos_lat = max(0.1, math.cos(lat_rad))
    lon = col_center_lon + (x * 15.0 / cos_lat)

    # Clamp lat/lon
    lat = max(-90.0, min(90.0, lat))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _get_direction_offset(r: int, d: int) -> Tuple[float, float]:
    """Computes planar offset (dx, dy) for direction d at resolution r."""
    if d == DIRECTION_CENTER:
        return 0.0, 0.0
    # Step size covers the base cell radius at r=1 and scales by 1/sqrt(7) per level
    scale = 1.0 / (math.sqrt(7.0) ** (r - 1))
    theta = math.radians(19.106 * r)
    alpha = (d - 1) * (math.pi / 3.0)
    angle = theta + alpha
    return math.cos(angle) * scale, math.sin(angle) * scale


def geo_to_h3(lat: float, lon: float, resolution: int) -> int:
    """
    Converts latitude and longitude coordinates into a 64-bit H3 cell index at the given resolution.
    """
    if not (0 <= resolution <= H3_MAX_RESOLUTION):
        raise ValueError(f"Resolution must be 0-{H3_MAX_RESOLUTION}")

    base_cell, target_x, target_y = _base_cell_for_lat_lon(lat, lon)
    digits: List[int] = []

    curr_x = 0.0
    curr_y = 0.0

    # Greedy closest-candidate selection at each resolution level
    for r in range(1, resolution + 1):
        best_d = 0
        best_dist_sq = float("inf")
        best_dx, best_dy = 0.0, 0.0

        for d in range(7):
            dx, dy = _get_direction_offset(r, d)
            cand_x = curr_x + dx
            cand_y = curr_y + dy
            dist_sq = (target_x - cand_x) ** 2 + (target_y - cand_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_d = d
                best_dx, best_dy = dx, dy

        digits.append(best_d)
        curr_x += best_dx
        curr_y += best_dy

    return encode_h3(resolution, base_cell, digits)


def h3_to_geo(h3_index: int) -> Tuple[float, float]:
    """
    Calculates the centroid (latitude, longitude) of an H3 cell.
    """
    res, base_cell, digits = decode_h3(h3_index)

    curr_x = 0.0
    curr_y = 0.0

    for r, d in enumerate(digits, start=1):
        dx, dy = _get_direction_offset(r, d)
        curr_x += dx
        curr_y += dy

    return _lat_lon_from_base_cell(base_cell, curr_x, curr_y)


def h3_to_geo_boundary(h3_index: int) -> List[Tuple[float, float]]:
    """
    Generates the 6 boundary vertices (lat, lon) forming the regular hexagon of the H3 cell.
    """
    center_lat, center_lon = h3_to_geo(h3_index)
    res = h3_get_resolution(h3_index)

    # Cell radius decreases exponentially by factor of sqrt(7) per resolution level
    base_radius_deg = 3.0 / (math.sqrt(7.0) ** res)

    vertices: List[Tuple[float, float]] = []
    for i in range(6):
        angle = math.radians(60.0 * i + 30.0)
        d_lat = base_radius_deg * math.sin(angle)
        cos_lat = max(0.1, math.cos(math.radians(center_lat)))
        d_lon = (base_radius_deg * math.cos(angle)) / cos_lat

        v_lat = max(-90.0, min(90.0, center_lat + d_lat))
        v_lon = ((center_lon + d_lon + 180.0) % 360.0) - 180.0
        vertices.append((v_lat, v_lon))

    return vertices


def k_ring(origin: int, k: int) -> Set[int]:
    """
    Finds all H3 cells within graph step distance k from origin (including origin).
    Explores the hexagonal grid topology.
    """
    if k < 0:
        return set()
    if k == 0:
        return {origin}

    res, base_cell, digits = decode_h3(origin)
    results: Set[int] = {origin}
    current_ring = {origin}

    for step in range(k):
        next_ring: Set[int] = set()
        for cell in current_ring:
            c_res, c_base, c_digits = decode_h3(cell)
            # Perturb the last digit or neighboring directions
            for d in range(7):
                new_digits = list(c_digits)
                if new_digits:
                    new_digits[-1] = d
                    neighbor = encode_h3(c_res, c_base, new_digits)
                    if neighbor not in results:
                        next_ring.add(neighbor)
                        results.add(neighbor)
        if not next_ring:
            break
        current_ring = next_ring

    return results


def h3_distance(origin: int, destination: int) -> int:
    """
    Approximates grid step distance between two H3 cells at the same resolution.
    """
    res_orig = h3_get_resolution(origin)
    res_dest = h3_get_resolution(destination)
    if res_orig != res_dest:
        raise ValueError("Distance only supported for cells at identical resolution")

    if origin == destination:
        return 0

    lat1, lon1 = h3_to_geo(origin)
    lat2, lon2 = h3_to_geo(destination)

    # Great circle distance
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    dist_km = EARTH_RADIUS_KM * c

    # Convert km distance to approximate cell edge step count at this resolution
    cell_edge_km = 1107.0 / (math.sqrt(7.0) ** res_orig)
    return max(1, int(round(dist_km / max(1e-5, cell_edge_km))))
