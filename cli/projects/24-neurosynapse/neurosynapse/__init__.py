"""NeuroSynapse: Neuromorphic Computing, Spiking Neural Network (SNN) & Event-Based Vision Engine.

Zero external dependencies. Pure Python standard library.
"""

from .dvs import (
    BackgroundActivityFilter,
    DVSEvent,
    DVSStimulusGenerator,
    DVSStream,
    OpticalFlowEstimator,
    RefractoryFilter,
    SurfaceOfActiveEvents,
)
from .encoding import (
    BernoulliEncoder,
    DeltaEncoder,
    ExponentialFilterDecoder,
    FirstSpikeWinnerDecoder,
    LatencyEncoder,
    PhaseEncoder,
    PoissonEncoder,
    RankOrderEncoder,
    RateDecoder,
    TTFSEncoder,
)
from .engine import (
    NeuromorphicEngine,
    SpikingAudioClassifier,
    UnsupervisedSTDPFeatureExtractor,
)
from .learning import (
    AdamOptimizer,
    ArcTanSurrogate,
    FastSigmoidSurrogate,
    SGDOptimizer,
    SpikingDenseLayer,
    SurrogateFunction,
    TriangularSurrogate,
    mean_squared_rate_loss,
)
from .network import (
    EventDrivenSimulator,
    LiquidStateMachine,
    SpikingNetwork,
)
from .neurons import (
    HodgkinHuxleyNeuron,
    IzhikevichNeuron,
    LIFNeuron,
    NeuronPopulation,
    SpikeEvent,
    SpikingNeuron,
)
from .synapses import (
    STDPConfig,
    Synapse,
    SynapseType,
    SynapticMatrix,
    TripletSTDPConfig,
    TripletSynapse,
)
from .visualizer import (
    BrailleCanvas,
    render_dvs_accumulator,
    render_oscilloscope,
    render_spike_raster,
    render_telemetry_hud,
)

__version__ = "1.0.0"
__all__ = [
    # Neurons
    "SpikingNeuron",
    "LIFNeuron",
    "IzhikevichNeuron",
    "HodgkinHuxleyNeuron",
    "NeuronPopulation",
    "SpikeEvent",
    # Synapses & STDP
    "Synapse",
    "SynapseType",
    "STDPConfig",
    "TripletSynapse",
    "TripletSTDPConfig",
    "SynapticMatrix",
    # Learning & BPTT
    "SurrogateFunction",
    "FastSigmoidSurrogate",
    "ArcTanSurrogate",
    "TriangularSurrogate",
    "SpikingDenseLayer",
    "SGDOptimizer",
    "AdamOptimizer",
    "mean_squared_rate_loss",
    # DVS & Vision
    "DVSEvent",
    "DVSStream",
    "RefractoryFilter",
    "BackgroundActivityFilter",
    "SurfaceOfActiveEvents",
    "OpticalFlowEstimator",
    "DVSStimulusGenerator",
    # Encoders & Decoders
    "PoissonEncoder",
    "BernoulliEncoder",
    "TTFSEncoder",
    "LatencyEncoder",
    "RankOrderEncoder",
    "PhaseEncoder",
    "DeltaEncoder",
    "RateDecoder",
    "ExponentialFilterDecoder",
    "FirstSpikeWinnerDecoder",
    # Network Topologies & Engines
    "SpikingNetwork",
    "EventDrivenSimulator",
    "LiquidStateMachine",
    # Visualizer
    "BrailleCanvas",
    "render_spike_raster",
    "render_oscilloscope",
    "render_dvs_accumulator",
    "render_telemetry_hud",
    # High-level Orchestration
    "NeuromorphicEngine",
    "SpikingAudioClassifier",
    "UnsupervisedSTDPFeatureExtractor",
]
