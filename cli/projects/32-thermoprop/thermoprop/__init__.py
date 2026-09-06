"""
ThermoProp: Compressible Gas Dynamics, Supersonic De Laval Nozzle & Rocket Propulsion Engine.

A pure Python 3.10+ standard library aerothermodynamics and rocket propulsion toolkit:
- 1D/2D Compressible Gas Dynamics (Isentropic, Rankine-Hugoniot shocks, Oblique shocks, Prandtl-Meyer fans)
- 2D Method of Characteristics (MOC) Minimum-Length Supersonic Nozzle (MLN) Contour Synthesis
- Ideal and Actual Rocket Propulsion Thermochemistry (c*, C_F, Isp, Tsiolkovsky delta-v)
- Regenerative Thrust Chamber Cooling & Bartz Convective Heat Transfer Equation
- Supersonic Exhaust Plume Adaptation, Summerfield Separation & Mach Diamond Shock Cells
- Sub-Pixel Unicode Braille Visualizer & Rocket Telemetry HUD
"""

from __future__ import annotations

from .cooling import (
    CoolantChannelProperties,
    RegenerativeCoolingEngine,
    ThermalStationResult,
    WallMaterial,
)
from .gas_dynamics import (
    CompressibleFlowEngine,
    GasProperties,
    IsentropicFlowState,
    NormalShockState,
    ObliqueShockState,
)
from .moc_nozzle import (
    CharacteristicPoint,
    MethodOfCharacteristicsNozzle,
    NozzleContour,
)
from .plume import (
    ExhaustPlumeEngine,
    MachDiamondCell,
    PlumeRegime,
    PlumeStructure,
)
from .propulsion import (
    PropellantData,
    PropellantLibrary,
    RocketEngineState,
    RocketPropulsionEngine,
    SEA_LEVEL_PRESSURE,
    STANDARD_GRAVITY,
)
from .visualizer import (
    BrailleCanvas,
    RocketVisualizer,
)

__version__ = "1.0.0"
__all__ = [
    # Gas Dynamics
    "GasProperties",
    "IsentropicFlowState",
    "NormalShockState",
    "ObliqueShockState",
    "CompressibleFlowEngine",
    # MOC Nozzle
    "CharacteristicPoint",
    "NozzleContour",
    "MethodOfCharacteristicsNozzle",
    # Propulsion
    "STANDARD_GRAVITY",
    "SEA_LEVEL_PRESSURE",
    "PropellantData",
    "PropellantLibrary",
    "RocketEngineState",
    "RocketPropulsionEngine",
    # Cooling & Heat Transfer
    "WallMaterial",
    "CoolantChannelProperties",
    "ThermalStationResult",
    "RegenerativeCoolingEngine",
    # Plume & Shock Structure
    "PlumeRegime",
    "MachDiamondCell",
    "PlumeStructure",
    "ExhaustPlumeEngine",
    # Visualizer
    "BrailleCanvas",
    "RocketVisualizer",
]
