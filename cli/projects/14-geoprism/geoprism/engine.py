"""
GeoPrism: Spatial Analytics & Geospatial Engine.
Implements Dual R*-Tree Synchronous Spatial Join, R*-Tree-accelerated DBSCAN clustering,
H3 hexagonal binning, and Kernel Density Estimation (KDE) rasterization.
"""

import math
from typing import List, Tuple, Dict, Set, Any, Optional
from geoprism.aabb import AABB
from geoprism.rtree import RStarTree, RStarNode
from geoprism.h3 import geo_to_h3


def spatial_join(tree_a: RStarTree, tree_b: RStarTree) -> List[Tuple[Any, Any]]:
    """
    Performs a synchronous dual-tree spatial join between two R*-Trees.
    Finds all pairs of items (item_a, item_b) whose bounding boxes intersect.
    Time Complexity: Substantially outperforms O(N x M) pairwise checking via hierarchical pruning.
    """
    if tree_a.size == 0 or tree_b.size == 0 or tree_a.root.aabb is None or tree_b.root.aabb is None:
        return []

    results: List[Tuple[Any, Any]] = []

    def _dual_traverse(node_a: RStarNode, node_b: RStarNode) -> None:
        if node_a.aabb is None or node_b.aabb is None or not node_a.aabb.intersects(node_b.aabb):
            return

        if node_a.is_leaf and node_b.is_leaf:
            for item_a, box_a in node_a.entries:
                for item_b, box_b in node_b.entries:
                    if box_a.intersects(box_b):
                        results.append((item_a, item_b))
        elif node_a.is_leaf:
            for child_b, box_b in node_b.entries:
                if node_a.aabb.intersects(box_b):
                    _dual_traverse(node_a, child_b)
        elif node_b.is_leaf:
            for child_a, box_a in node_a.entries:
                if box_a.intersects(node_b.aabb):
                    _dual_traverse(child_a, node_b)
        else:
            for child_a, box_a in node_a.entries:
                for child_b, box_b in node_b.entries:
                    if box_a.intersects(box_b):
                        _dual_traverse(child_a, child_b)

    _dual_traverse(tree_a.root, tree_b.root)
    return results


def dbscan(
    points: List[Tuple[float, ...]],
    eps: float,
    min_pts: int
) -> Dict[int, List[int]]:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    accelerated by R*-Tree spatial range queries.
    Parameters:
      - points: N-dimensional coordinate tuples
      - eps: Maximum neighborhood distance
      - min_pts: Minimum points to form a dense core region
    Returns:
      Dictionary mapping cluster_id to list of point indices.
      Cluster ID -1 represents noise points.
    """
    if not points:
        return {}

    dims = len(points[0])
    tree = RStarTree(max_entries=16, dimensions=dims)
    for idx, p in enumerate(points):
        tree.insert(idx, AABB.from_point(p))

    UNVISITED = -2
    NOISE = -1
    labels = [UNVISITED] * len(points)
    cluster_id = 0

    def _euclidean_dist(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

    def _region_query(point_idx: int) -> List[int]:
        pt = points[point_idx]
        cand = tree.search(AABB.from_point(pt, radius=eps))
        neighbors: List[int] = []
        for other_idx, _ in cand:
            if _euclidean_dist(pt, points[other_idx]) <= eps:
                neighbors.append(other_idx)
        return neighbors

    for i in range(len(points)):
        if labels[i] != UNVISITED:
            continue

        neighbors = _region_query(i)
        if len(neighbors) < min_pts:
            labels[i] = NOISE
        else:
            labels[i] = cluster_id
            queue = list(neighbors)
            seen = set(neighbors)

            head = 0
            while head < len(queue):
                q = queue[head]
                head += 1

                if labels[q] == NOISE:
                    labels[q] = cluster_id
                if labels[q] != UNVISITED:
                    continue

                labels[q] = cluster_id
                q_neighbors = _region_query(q)
                if len(q_neighbors) >= min_pts:
                    for nb in q_neighbors:
                        if nb not in seen:
                            seen.add(nb)
                            queue.append(nb)

            cluster_id += 1

    # Group point indices by cluster label
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)

    return clusters


def h3_hexbin(
    coords: List[Tuple[float, float]],
    resolution: int = 7
) -> Dict[int, int]:
    """
    Aggregates geographic (lat, lon) coordinates into discrete H3 hexagonal cells.
    Returns mapping from H3 cell index to point occurrence count.
    """
    bins: Dict[int, int] = {}
    for lat, lon in coords:
        cell = geo_to_h3(lat, lon, resolution)
        bins[cell] = bins.get(cell, 0) + 1
    return bins


def spatial_kde_raster(
    points: List[Tuple[float, float]],
    grid_w: int = 60,
    grid_h: int = 25,
    bandwidth: Optional[float] = None
) -> Tuple[List[List[float]], AABB]:
    """
    Computes a 2D Gaussian Kernel Density Estimation (KDE) raster grid over points.
    Returns:
      (matrix[height][width] normalized to [0.0, 1.0], bounding_box)
    """
    if not points:
        return [[0.0] * grid_w for _ in range(grid_h)], AABB((0, 0), (1, 1))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add margin
    dx = max(max_x - min_x, 1e-3)
    dy = max(max_y - min_y, 1e-3)
    pad_x = dx * 0.05
    pad_y = dy * 0.05
    bbox = AABB((min_x - pad_x, min_y - pad_y), (max_x + pad_x, max_y + pad_y))

    if bandwidth is None:
        # Silverman's rule of thumb approximation
        bandwidth = max(0.5 * (dx + dy) / math.sqrt(len(points)), 1e-3)

    two_h_sq = 2.0 * bandwidth * bandwidth

    grid: List[List[float]] = []
    max_density = 0.0

    step_x = (bbox.upper[0] - bbox.lower[0]) / max(1, grid_w - 1)
    step_y = (bbox.upper[1] - bbox.lower[1]) / max(1, grid_h - 1)

    for row in range(grid_h):
        # Y axis increases upwards, so row 0 is top (max_y)
        gy = bbox.upper[1] - row * step_y
        row_vals: List[float] = []
        for col in range(grid_w):
            gx = bbox.lower[0] + col * step_x
            density = 0.0
            for px, py in points:
                dist_sq = (gx - px) ** 2 + (gy - py) ** 2
                density += math.exp(-dist_sq / two_h_sq)
            if density > max_density:
                max_density = density
            row_vals.append(density)
        grid.append(row_vals)

    # Normalize to [0.0, 1.0]
    if max_density > 0.0:
        for r in range(grid_h):
            for c in range(grid_w):
                grid[r][c] /= max_density

    return grid, bbox
