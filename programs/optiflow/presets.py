"""
OptiFlow 2D: Curated Aerodynamics & CFD Presets
Zero external dependencies. Pure Python standard library.
"""

from typing import Dict, Any


AERO_PRESETS: Dict[str, Dict[str, Any]] = {
    "naca0012": {
        "name": "NACA 0012 Symmetric Airfoil",
        "description": "Standard symmetric NACA 4-digit airfoil profile at moderate angle of attack.",
        "type": "naca",
        "m": 0.0,
        "p": 0.0,
        "t": 0.12,
        "chord": 0.55,
        "alpha": 6.0,
        "reynolds": 450.0,
        "u_inf": 1.0,
    },
    "naca4412": {
        "name": "NACA 4412 High-Lift Cambered Airfoil",
        "description": "Asymmetric cambered airfoil profile with high lift generation.",
        "type": "naca",
        "m": 0.04,
        "p": 0.4,
        "t": 0.12,
        "chord": 0.55,
        "alpha": 4.0,
        "reynolds": 500.0,
        "u_inf": 1.0,
    },
    "naca2412_stall": {
        "name": "NACA 2412 Near-Stall (Alpha = 16 deg)",
        "description": "High angle of attack inducing boundary layer separation and stall turbulence.",
        "type": "naca",
        "m": 0.02,
        "p": 0.4,
        "t": 0.12,
        "chord": 0.50,
        "alpha": 16.0,
        "reynolds": 350.0,
        "u_inf": 1.0,
    },
    "cylinder_karman": {
        "name": "Circular Cylinder (Von Karman Vortex Street)",
        "description": "Periodic alternating vortex shedding in subcritical Reynolds laminar wake.",
        "type": "cylinder",
        "radius": 0.075,
        "reynolds": 250.0,
        "u_inf": 1.0,
    },
    "venturi_nozzle": {
        "name": "Venturi Convergent-Divergent Nozzle",
        "description": "Constricted throat demonstrating Bernoulli acceleration and pressure drop.",
        "type": "venturi",
        "throat": 0.35,
        "inlet": 0.75,
        "reynolds": 600.0,
        "u_inf": 1.0,
    },
    "backward_step": {
        "name": "Backward-Facing Step Flow",
        "description": "Sudden geometric expansion creating flow separation and recirculation eddy.",
        "type": "step",
        "step_x": 0.45,
        "step_h": 0.35,
        "reynolds": 300.0,
        "u_inf": 1.0,
    }
}
