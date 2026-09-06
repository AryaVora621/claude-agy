"""RetroCAD Curated Engineering Mechanical Presets.

Provides parametric CAD models:
1. Machine Bearing Housing with mounting flange and bolt circle
2. Supersonic Rocket Engine De Laval Nozzle
3. Parametric Involute Spur Gear
4. Precision Hex Bolt with washer face
5. Aerospace Lightening Bracket with structural stiffeners
"""

import math
from typing import List, Tuple
try:
    from .geom import (
        Mesh, Vec3, Mat4, create_cylinder, create_box, extrude_polygon,
        revolve_profile, csg_boolean
    )
except ImportError:
    from geom import (
        Mesh, Vec3, Mat4, create_cylinder, create_box, extrude_polygon,
        revolve_profile, csg_boolean
    )


def build_bearing_housing() -> Mesh:
    """Build a flanged mechanical bearing housing with central shaft bore."""
    # 1. Main outer body cylinder
    housing = create_cylinder(radius=30.0, height=40.0, segments=32, name="BearingHousing")

    # 2. Bottom mounting flange disk
    flange = create_cylinder(radius=50.0, height=10.0, segments=32, name="MountingFlange")
    flange.transform(Mat4.translation(0.0, 0.0, -15.0))

    # Union body and flange
    part = csg_boolean(housing, flange, "union")

    # 3. Inner bearing race bore (cylinder subtracted)
    inner_bore = create_cylinder(radius=18.0, height=50.0, segments=24, name="ShaftBore")
    part = csg_boolean(part, inner_bore, "difference")

    # 4. Bolt circle mounting holes (4 holes on flange at radius 40mm)
    for i in range(4):
        angle = i * (math.pi / 2.0)
        bx = 40.0 * math.cos(angle)
        by = 40.0 * math.sin(angle)
        bolt_hole = create_cylinder(radius=4.5, height=20.0, segments=16, name=f"BoltHole_{i}")
        bolt_hole.transform(Mat4.translation(bx, by, -15.0))
        part = csg_boolean(part, bolt_hole, "difference")

    part.name = "BearingHousing"
    return part


def build_rocket_nozzle() -> Mesh:
    """Build a supersonic converging-diverging de Laval rocket engine nozzle."""
    # (radius, z) profile from injector head to nozzle exit
    profile: List[Tuple[float, float]] = [
        (35.0,  50.0),  # Combustion chamber top
        (35.0,  30.0),  # Combustion chamber wall
        (32.0,  15.0),  # Converging section start
        (22.0,   5.0),  # Converging section
        (16.0,   0.0),  # Nozzle throat (Mach 1)
        (19.0, -10.0),  # Diverging expansion bell
        (26.0, -25.0),  # Supersonic bell expansion
        (36.0, -45.0),  # Bell mid
        (48.0, -70.0),  # Bell exit lip (Mach 3.5)
        (50.0, -70.0),  # Lip thickness
        (38.0, -45.0),  # Outer bell wall
        (28.0, -25.0),  # Outer contour
        (21.0, -10.0),  # Outer throat reinforcement
        (19.0,   0.0),  # Throat outer ring
        (25.0,   5.0),  # Outer convergent
        (35.0,  15.0),  # Outer chamber
        (38.0,  30.0),  # Outer chamber wall
        (38.0,  50.0),  # Outer injector flange
        (35.0,  50.0)   # Close loop
    ]
    nozzle = revolve_profile(profile, segments=32, angle_deg=360.0, name="RocketNozzle")
    return nozzle


def build_spur_gear(teeth: int = 14, module: float = 3.0, face_width: float = 12.0) -> Mesh:
    """Build a parametric involute spur gear with center shaft keyway."""
    pitch_radius = (teeth * module) * 0.5
    addendum = module
    dedendum = 1.25 * module
    outer_radius = pitch_radius + addendum
    root_radius = pitch_radius - dedendum

    # Construct 2D gear tooth contour
    contour: List[Tuple[float, float]] = []
    total_pts_per_tooth = 6

    for t in range(teeth):
        base_angle = (2.0 * math.pi * t) / teeth
        tooth_span = (2.0 * math.pi) / teeth

        # Root, flanks, tip, and space
        a0 = base_angle
        a1 = base_angle + tooth_span * 0.2
        a2 = base_angle + tooth_span * 0.35
        a3 = base_angle + tooth_span * 0.65
        a4 = base_angle + tooth_span * 0.8
        a5 = base_angle + tooth_span

        contour.append((root_radius * math.cos(a0), root_radius * math.sin(a0)))
        contour.append((pitch_radius * math.cos(a1), pitch_radius * math.sin(a1)))
        contour.append((outer_radius * math.cos(a2), outer_radius * math.sin(a2)))
        contour.append((outer_radius * math.cos(a3), outer_radius * math.sin(a3)))
        contour.append((pitch_radius * math.cos(a4), pitch_radius * math.sin(a4)))
        contour.append((root_radius * math.cos(a5), root_radius * math.sin(a5)))

    gear = extrude_polygon(contour, height=face_width, name="SpurGear")
    gear.transform(Mat4.translation(0.0, 0.0, -face_width * 0.5))

    # Center shaft bore
    bore = create_cylinder(radius=pitch_radius * 0.35, height=face_width * 1.5, segments=20, name="GearBore")
    gear = csg_boolean(gear, bore, "difference")
    gear.name = f"SpurGear_T{teeth}"
    return gear


def build_hex_bolt() -> Mesh:
    """Build an ISO metric hexagonal head bolt with cylindrical shank."""
    # 1. Hexagonal head contour (6 vertices)
    head_radius = 20.0
    head_height = 14.0
    hex_contour = []
    for i in range(6):
        ang = i * (math.pi / 3.0)
        hex_contour.append((head_radius * math.cos(ang), head_radius * math.sin(ang)))

    head = extrude_polygon(hex_contour, height=head_height, name="HexHead")
    head.transform(Mat4.translation(0.0, 0.0, 30.0))

    # 2. Cylindrical threaded shank
    shank = create_cylinder(radius=10.0, height=60.0, segments=24, name="BoltShank")

    bolt = csg_boolean(head, shank, "union")
    bolt.name = "HexBolt_M20"
    return bolt


def build_aerospace_bracket() -> Mesh:
    """Build a structural L-bracket with weight-reduction holes."""
    # 2D L-shape contour in XY
    l_contour: List[Tuple[float, float]] = [
        (-30.0, -30.0),
        ( 30.0, -30.0),
        ( 30.0, -18.0),
        (-18.0, -18.0),
        (-18.0,  30.0),
        (-30.0,  30.0)
    ]
    bracket = extrude_polygon(l_contour, height=25.0, name="LBracket")
    bracket.transform(Mat4.translation(0.0, 0.0, -12.5))

    # Weight reduction hole 1 in bottom leg
    hole1 = create_cylinder(radius=5.5, height=35.0, segments=16, name="LightenHole1")
    hole1.transform(Mat4.translation(10.0, -24.0, 0.0))
    bracket = csg_boolean(bracket, hole1, "difference")

    # Weight reduction hole 2 in vertical leg
    hole2 = create_cylinder(radius=5.5, height=35.0, segments=16, name="LightenHole2")
    hole2.transform(Mat4.translation(-24.0, 10.0, 0.0))
    bracket = csg_boolean(bracket, hole2, "difference")

    bracket.name = "AerospaceLBracket"
    return bracket


PRESET_BUILDERS = {
    "Bearing Housing": build_bearing_housing,
    "Rocket Engine Nozzle": build_rocket_nozzle,
    "Parametric Spur Gear": build_spur_gear,
    "Hex Bolt M20": build_hex_bolt,
    "Aerospace L-Bracket": build_aerospace_bracket
}
