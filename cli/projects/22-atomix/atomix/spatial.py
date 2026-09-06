"""
Atomix: Spatial Partitioning, Periodic Boundaries, and Neighbor Search.
Implements:
  - 3D Linked Cell Lists for O(N) neighbor search
  - Forward half-neighborhood 13-cell neighbor stencil
  - Verlet neighbor skin lists with displacement-triggered rebuilds
  - Minimum image convention pair distance evaluation
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Callable, Set

from atomix.types import Vector3D, Atom, SimulationBox


class LinkedCellList:
    """
    3D Linked Cell spatial partitioner for O(N) pair interaction calculation.
    Discretizes the simulation box into subcells of dimension >= cutoff.
    """
    def __init__(self, box: SimulationBox, cutoff: float) -> None:
        if cutoff <= 0.0:
            raise ValueError(f"Cutoff must be positive, got {cutoff}")
        self.box = box
        self.cutoff = float(cutoff)
        self.cutoff_sq = self.cutoff * self.cutoff

        # Number of cells along each axis (must be at least 1)
        self.nx = max(1, int(math.floor(self.box.lx / self.cutoff)))
        self.ny = max(1, int(math.floor(self.box.ly / self.cutoff)))
        self.nz = max(1, int(math.floor(self.box.lz / self.cutoff)))

        # Actual cell dimensions (guaranteed >= cutoff)
        self.cell_x = self.box.lx / self.nx
        self.cell_y = self.box.ly / self.ny
        self.cell_z = self.box.lz / self.nz
        self.inv_cell_x = 1.0 / self.cell_x
        self.inv_cell_y = 1.0 / self.cell_y
        self.inv_cell_z = 1.0 / self.cell_z

        self.num_cells = self.nx * self.ny * self.nz

        # Precompute forward neighbor cell offsets (13 forward directions)
        # Prevents double counting of pairs: (dx, dy, dz)
        self.forward_offsets: List[Tuple[int, int, int]] = []
        # dz = +1: all 9 cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self.forward_offsets.append((dx, dy, 1))
        # dz = 0: dy = +1 (3 cells)
        for dx in (-1, 0, 1):
            self.forward_offsets.append((dx, 1, 0))
        # dz = 0, dy = 0: dx = +1 (1 cell)
        self.forward_offsets.append((1, 0, 0))

    def _cell_coords(self, pos: Vector3D) -> Tuple[int, int, int]:
        """Convert a 3D coordinate into integer cell coordinates (cx, cy, cz)."""
        wrapped = self.box.wrap_position(pos)
        cx = int(wrapped.x * self.inv_cell_x)
        cy = int(wrapped.y * self.inv_cell_y)
        cz = int(wrapped.z * self.inv_cell_z)

        # Guard against edge floating point roundoff
        if cx >= self.nx:
            cx = self.nx - 1
        elif cx < 0:
            cx = 0
        if cy >= self.ny:
            cy = self.ny - 1
        elif cy < 0:
            cy = 0
        if cz >= self.nz:
            cz = self.nz - 1
        elif cz < 0:
            cz = 0

        return (cx, cy, cz)

    def _cell_index(self, cx: int, cy: int, cz: int) -> int:
        """Flatten 3D cell coordinates into a single 1D array index."""
        return (cz * self.ny + cy) * self.nx + cx

    def build(self, atoms: List[Atom]) -> List[List[int]]:
        """
        Partition all atoms into spatial cells.
        Returns a list of atom index lists for each cell.
        Runtime complexity: strictly O(N).
        """
        cells: List[List[int]] = [[] for _ in range(self.num_cells)]
        for idx, atom in enumerate(atoms):
            cx, cy, cz = self._cell_coords(atom.position)
            c_idx = self._cell_index(cx, cy, cz)
            cells[c_idx].append(idx)
        return cells

    def find_all_pairs(
        self,
        atoms: List[Atom],
        excluded_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int, Vector3D, float]]:
        """
        Extract all atom pairs within the cutoff radius.
        Returns list of (atom1_idx, atom2_idx, r12_min_image_vector, distance_sq).
        guarantees atom1_idx < atom2_idx.
        """
        cells = self.build(atoms)
        pairs: List[Tuple[int, int, Vector3D, float]] = []

        # If system is small (fewer than 3 cells along any axis), fallback to direct loop
        # to avoid self-neighbor aliases across periodic boundaries
        if self.nx < 3 or self.ny < 3 or self.nz < 3:
            n = len(atoms)
            for i in range(n):
                pos_i = atoms[i].position
                for j in range(i + 1, n):
                    if excluded_pairs and ((i, j) in excluded_pairs or (j, i) in excluded_pairs):
                        continue
                    r12 = self.box.minimum_image_vector(pos_i, atoms[j].position)
                    d_sq = r12.norm_sq()
                    if d_sq < self.cutoff_sq:
                        pairs.append((i, j, r12, d_sq))
            return pairs

        # Standard O(N) linked cell traversal
        for cz in range(self.nz):
            for cy in range(self.ny):
                for cx in range(self.nx):
                    c_idx = self._cell_index(cx, cy, cz)
                    current_cell = cells[c_idx]

                    # 1. Intra-cell pairs (all pairs within the same cell)
                    n_in_cell = len(current_cell)
                    for i_idx in range(n_in_cell):
                        idx_i = current_cell[i_idx]
                        pos_i = atoms[idx_i].position
                        for j_idx in range(i_idx + 1, n_in_cell):
                            idx_j = current_cell[j_idx]
                            first = min(idx_i, idx_j)
                            second = max(idx_i, idx_j)
                            if excluded_pairs and (first, second) in excluded_pairs:
                                continue
                            r12 = self.box.minimum_image_vector(pos_i, atoms[idx_j].position)
                            d_sq = r12.norm_sq()
                            if d_sq < self.cutoff_sq:
                                pairs.append((first, second, r12, d_sq))

                    # 2. Inter-cell pairs with the 13 forward neighbors
                    for ox, oy, oz in self.forward_offsets:
                        ncx = (cx + ox) % self.nx
                        ncy = (cy + oy) % self.ny
                        ncz = (cz + oz) % self.nz
                        neigh_idx = self._cell_index(ncx, ncy, ncz)
                        neigh_cell = cells[neigh_idx]

                        for idx_i in current_cell:
                            pos_i = atoms[idx_i].position
                            for idx_j in neigh_cell:
                                first = min(idx_i, idx_j)
                                second = max(idx_i, idx_j)
                                if excluded_pairs and (first, second) in excluded_pairs:
                                    continue
                                r12 = self.box.minimum_image_vector(pos_i, atoms[idx_j].position)
                                d_sq = r12.norm_sq()
                                if d_sq < self.cutoff_sq:
                                    pairs.append((first, second, r12, d_sq))

        return pairs


class VerletNeighborList:
    """
    Verlet neighbor list with skin buffer (r_list = cutoff + skin).
    Drastically reduces pair distance computations by checking neighbor candidate lists
    and rebuilding only when cumulative atomic displacement exceeds skin / 2.
    """
    def __init__(
        self,
        box: SimulationBox,
        cutoff: float,
        skin: float = 0.3,
    ) -> None:
        self.box = box
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.r_list = self.cutoff + self.skin
        self.r_list_sq = self.r_list * self.r_list
        self.cutoff_sq = self.cutoff * self.cutoff

        self.cell_partitioner = LinkedCellList(box, self.r_list)
        self.neighbors: Dict[int, List[int]] = {}
        self.reference_positions: List[Vector3D] = []

    def needs_rebuild(self, atoms: List[Atom]) -> bool:
        """
        Check if maximum particle displacement since last rebuild exceeds half skin.
        If sum of two largest displacements >= skin, a particle outside r_list could enter r_cutoff.
        """
        if not self.reference_positions or len(self.reference_positions) != len(atoms):
            return True

        max_d1 = 0.0
        max_d2 = 0.0
        for i, atom in enumerate(atoms):
            # Displacement without boundary wrap to track total path traveled
            disp_vec = atom.position - self.reference_positions[i]
            d = disp_vec.norm()
            if d > max_d1:
                max_d2 = max_d1
                max_d1 = d
            elif d > max_d2:
                max_d2 = d

        return (max_d1 + max_d2) >= self.skin

    def rebuild(
        self,
        atoms: List[Atom],
        excluded_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """
        Reconstruct the neighbor lists for all atoms within r_list.
        """
        self.neighbors = {i: [] for i in range(len(atoms))}
        self.reference_positions = [atom.position for atom in atoms]

        # Use cell partitioner at extended skin radius
        pairs = self.cell_partitioner.find_all_pairs(atoms, excluded_pairs)
        for i, j, _, d_sq in pairs:
            if d_sq < self.r_list_sq:
                self.neighbors[i].append(j)

    def get_active_pairs(
        self,
        atoms: List[Atom],
        excluded_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int, Vector3D, float]]:
        """
        Evaluate pairs from the cached Verlet neighbor list within the true physical cutoff.
        Rebuilds automatically if particles have migrated significantly.
        """
        if self.needs_rebuild(atoms):
            self.rebuild(atoms, excluded_pairs)

        active_pairs: List[Tuple[int, int, Vector3D, float]] = []
        for i, neighbor_indices in self.neighbors.items():
            pos_i = atoms[i].position
            for j in neighbor_indices:
                r12 = self.box.minimum_image_vector(pos_i, atoms[j].position)
                d_sq = r12.norm_sq()
                if d_sq < self.cutoff_sq:
                    active_pairs.append((i, j, r12, d_sq))

        return active_pairs
