"""
Curated Library of 2D Structural Engineering FEA Benchmark Models.
Contains Truss, CST, and Quad4 discretizations for classical structural problems.
"""

import math
from typing import Dict, Callable

try:
    from .fea_engine import FEAModel
except ImportError:
    from fea_engine import FEAModel


def build_warren_truss() -> FEAModel:
    """
    Warren-Pratt Pin-Jointed Bridge Truss:
    7-Bay truss with top and bottom chords, vertical posts, and alternating diagonals.
    Pinned at Node 0, roller at Node 6, deck truck loads applied to bottom nodes.
    """
    model = FEAModel("Warren-Pratt Bridge Truss")
    bay_length = 3.0   # meters
    height = 2.5       # meters
    E = 210e9          # Steel: 210 GPa
    A_chord = 0.005    # 50 cm^2
    A_web = 0.003      # 30 cm^2

    # Bottom chord nodes (y = 0): 0 to 6
    for i in range(7):
        is_pin = (i == 0)
        is_roller = (i == 6)
        fy = -50e3 if (1 <= i <= 5) else 0.0  # 50 kN downward vehicle load per internal deck node
        model.add_node(i, x=i * bay_length, y=0.0, fix_x=is_pin, fix_y=(is_pin or is_roller), fy=fy)

    # Top chord nodes (y = height): 7 to 12
    for i in range(6):
        node_id = 7 + i
        model.add_node(node_id, x=(i + 0.5) * bay_length, y=height, fix_x=False, fix_y=False)

    elem_id = 0
    # Bottom chord members
    for i in range(6):
        model.add_truss(elem_id, i, i + 1, area=A_chord, E=E)
        elem_id += 1

    # Top chord members
    for i in range(5):
        model.add_truss(elem_id, 7 + i, 7 + i + 1, area=A_chord, E=E)
        elem_id += 1

    # Web diagonals and verticals
    for i in range(6):
        top_node = 7 + i
        # Diagonal from bottom i to top i
        model.add_truss(elem_id, i, top_node, area=A_web, E=E)
        elem_id += 1
        # Diagonal from top i to bottom i+1
        model.add_truss(elem_id, top_node, i + 1, area=A_web, E=E)
        elem_id += 1

    return model


def build_cantilever_cst() -> FEAModel:
    """
    Cantilever Beam Discretized with Constant Strain Triangles (CST):
    Length L = 4.0 m, Depth H = 0.8 m, Thickness t = 0.1 m.
    Fixed left boundary (x = 0), downward shear load P = 100 kN applied at tip (x = L).
    """
    model = FEAModel("Cantilever Beam (CST Discretization)")
    L = 4.0
    H = 0.8
    nx = 8   # 8 segments along length
    ny = 3   # 3 segments along height
    dx = L / nx
    dy = H / ny
    E = 200e9
    nu = 0.28
    thickness = 0.1

    # Generate grid nodes
    node_map = {}
    node_id = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy - H / 2.0  # Centered vertically around neutral axis
            is_fixed = (i == 0)
            # Apply tip load shared among rightmost nodes
            fy = -100e3 / (ny + 1) if (i == nx) else 0.0
            model.add_node(node_id, x, y, fix_x=is_fixed, fix_y=is_fixed, fy=fy)
            node_map[(i, j)] = node_id
            node_id += 1

    # Generate 2 CST elements per rectangular cell
    elem_id = 0
    for j in range(ny):
        for i in range(nx):
            n_bl = node_map[(i, j)]
            n_br = node_map[(i + 1, j)]
            n_tr = node_map[(i + 1, j + 1)]
            n_tl = node_map[(i, j + 1)]

            # Lower triangle (BL -> BR -> TL)
            model.add_cst(elem_id, n_bl, n_br, n_tl, thickness=thickness, E=E, nu=nu)
            elem_id += 1

            # Upper triangle (BR -> TR -> TL)
            model.add_cst(elem_id, n_br, n_tr, n_tl, thickness=thickness, E=E, nu=nu)
            elem_id += 1

    return model


def build_kirsch_plate_hole() -> FEAModel:
    """
    Kirsch Stress Concentration Problem:
    Plate with central circular hole under horizontal uniaxial tension.
    Due to quadrant symmetry (top-right quarter modeled):
    x = 0 has roller (fix_x = True), y = 0 has roller (fix_y = True).
    Tensile pull on right edge (x = W).
    Displays classical theoretical stress concentration factor Kt approx 3.0 at hole rim.
    """
    model = FEAModel("Kirsch Plate with Circular Hole")
    R = 0.5   # Hole radius
    W = 2.0   # Quarter plate width
    H = 2.0   # Quarter plate height
    nr = 6    # Radial rings
    nth = 8   # Circumferential sectors (0 to 90 degrees)
    E = 210e9
    nu = 0.30
    thickness = 0.05
    tensile_stress = 10e6  # 10 MPa tension

    node_grid = {}
    node_id = 0

    for i in range(nr + 1):
        r_frac = i / nr
        # Geometric grading for finer mesh near hole edge
        radius = R + (W - R) * (r_frac ** 2.0)

        for j in range(nth + 1):
            theta = (math.pi / 2.0) * (j / nth)
            # Morph from circular arc to square outer boundary
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            x_circ = radius * cos_t
            y_circ = radius * sin_t

            # Blend towards outer boundary
            x_box = W * (radius / W) * (cos_t / max(cos_t, sin_t, 1e-4))
            y_box = H * (radius / H) * (sin_t / max(cos_t, sin_t, 1e-4))

            w_blend = r_frac
            x = (1.0 - w_blend) * x_circ + w_blend * min(W, max(0.0, x_box))
            y = (1.0 - w_blend) * y_circ + w_blend * min(H, max(0.0, y_box))

            # Symmetry boundary conditions:
            # Bottom edge (theta = 0, y = 0): fix_y = True
            # Left edge (theta = pi/2, x = 0): fix_x = True
            is_sym_x = (abs(x) < 1e-4 or j == nth)
            is_sym_y = (abs(y) < 1e-4 or j == 0)

            # Tensile load on right boundary (i == nr)
            fx = 0.0
            if i == nr:
                dy = (H / nth) if (0 < j < nth) else (0.5 * H / nth)
                fx = tensile_stress * thickness * dy

            model.add_node(node_id, x, y, fix_x=is_sym_x, fix_y=is_sym_y, fx=fx)
            node_grid[(i, j)] = node_id
            node_id += 1

    # Connect nodes into Quad4 elements
    elem_id = 0
    for i in range(nr):
        for j in range(nth):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            model.add_quad4(elem_id, n1, n2, n3, n4, thickness=thickness, E=E, nu=nu)
            elem_id += 1

    return model


def build_l_bracket() -> FEAModel:
    """
    L-Shaped Structural Bracket with Re-entrant Corner:
    Vertical leg fixed at top, horizontal leg loaded with vertical tip shear.
    Demonstrates localized high stress concentration at re-entrant inner corner.
    """
    model = FEAModel("L-Shaped Structural Bracket")
    thickness = 0.02  # 20 mm thick
    E = 70e9          # Aluminum: 70 GPa
    nu = 0.33

    # Generate 6x6 grid where top-right 3x3 region is void
    N = 6
    grid = {}
    node_id = 0

    for j in range(N + 1):
        for i in range(N + 1):
            # If in the upper-right quadrant (i > 3 and j > 3), skip (void region)
            if i > 3 and j > 3:
                continue

            x = i * 0.15   # 0 to 0.9 m
            y = j * 0.15   # 0 to 0.9 m

            # Fixed at top edge of vertical leg: j == N (y = 0.9, x <= 0.45)
            is_fixed = (j == N and i <= 3)

            # Downward point load at tip of horizontal leg: i == N, j == 0
            fy = -15e3 if (i == N and j <= 1) else 0.0

            model.add_node(node_id, x, y, fix_x=is_fixed, fix_y=is_fixed, fy=fy)
            grid[(i, j)] = node_id
            node_id += 1

    # Connect CST elements
    elem_id = 0
    for j in range(N):
        for i in range(N):
            if (i, j) in grid and (i+1, j) in grid and (i+1, j+1) in grid and (i, j+1) in grid:
                n_bl = grid[(i, j)]
                n_br = grid[(i + 1, j)]
                n_tr = grid[(i + 1, j + 1)]
                n_tl = grid[(i, j + 1)]

                model.add_cst(elem_id, n_bl, n_br, n_tl, thickness=thickness, E=E, nu=nu)
                elem_id += 1
                model.add_cst(elem_id, n_br, n_tr, n_tl, thickness=thickness, E=E, nu=nu)
                elem_id += 1

    return model


def build_thick_cylinder() -> FEAModel:
    """
    Thick-Walled Pressure Cylinder under Internal Hydraulic Pressure:
    Quarter-symmetric model (inner radius Ri = 0.4 m, outer radius Ro = 0.9 m).
    Matches analytical Lamé radial and circumferential hoop stress solutions.
    """
    model = FEAModel("Thick-Walled Cylinder (Lamé Benchmark)")
    Ri = 0.4
    Ro = 0.9
    nr = 5
    nth = 8
    thickness = 0.1
    P_int = 20e6   # 20 MPa internal pressure
    E = 205e9
    nu = 0.29

    grid = {}
    node_id = 0

    for i in range(nr + 1):
        r = Ri + (Ro - Ri) * (i / nr)
        for j in range(nth + 1):
            theta = (math.pi / 2.0) * (j / nth)
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # Symmetry: theta = 0 (y = 0) -> fix_y = True
            # Symmetry: theta = pi/2 (x = 0) -> fix_x = True
            fix_x = (j == nth)
            fix_y = (j == 0)

            # Radial pressure load applied to inner boundary (i == 0)
            fx = 0.0
            fy = 0.0
            if i == 0:
                arc_len = Ri * (math.pi / 2.0) / nth
                total_force = P_int * arc_len * thickness
                fx = total_force * math.cos(theta)
                fy = total_force * math.sin(theta)

            model.add_node(node_id, x, y, fix_x=fix_x, fix_y=fix_y, fx=fx, fy=fy)
            grid[(i, j)] = node_id
            node_id += 1

    # Connect Quad4 elements
    elem_id = 0
    for i in range(nr):
        for j in range(nth):
            n1 = grid[(i, j)]
            n2 = grid[(i + 1, j)]
            n3 = grid[(i + 1, j + 1)]
            n4 = grid[(i, j + 1)]
            model.add_quad4(elem_id, n1, n2, n3, n4, thickness=thickness, E=E, nu=nu)
            elem_id += 1

    return model


def build_quad4_shear_wall() -> FEAModel:
    """
    Multi-Story Shear Wall under Lateral Lateral Wind/Seismic Force:
    Height = 6.0 m, Width = 2.4 m, Thickness = 0.2 m.
    Discretized with Quad4 continuum elements, base completely clamped.
    """
    model = FEAModel("Multi-Story Shear Wall (Quad4)")
    W = 2.4
    H = 6.0
    nx = 4
    ny = 10
    dx = W / nx
    dy = H / ny
    E = 30e9     # Concrete: 30 GPa
    nu = 0.20
    thickness = 0.2

    grid = {}
    node_id = 0

    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            # Base fully fixed (j == 0)
            is_base = (j == 0)

            # Lateral horizontal loads increasing with height (seismic inverted triangular distribution)
            fx = 15e3 * (j / ny) if (i == 0 and j > 0) else 0.0

            model.add_node(node_id, x, y, fix_x=is_base, fix_y=is_base, fx=fx, fy=0.0)
            grid[(i, j)] = node_id
            node_id += 1

    elem_id = 0
    for j in range(ny):
        for i in range(nx):
            n1 = grid[(i, j)]
            n2 = grid[(i + 1, j)]
            n3 = grid[(i + 1, j + 1)]
            n4 = grid[(i, j + 1)]
            model.add_quad4(elem_id, n1, n2, n3, n4, thickness=thickness, E=E, nu=nu)
            elem_id += 1

    return model


PRESETS: Dict[str, Callable[[], FEAModel]] = {
    "warren_truss": build_warren_truss,
    "cantilever_cst": build_cantilever_cst,
    "kirsch_plate_hole": build_kirsch_plate_hole,
    "l_bracket": build_l_bracket,
    "thick_cylinder": build_thick_cylinder,
    "quad4_shear_wall": build_quad4_shear_wall
}
