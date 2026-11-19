"""Simulate 3D electrostatic potential volumes from atomic structures in PyTorch.

Built on the teamtomo primitives ``torch-calculate-electrostatic-potential``,
``torch-structure-manipulation``, ``torch-fourier-rescale``, and
``torch-fourier-filter``. Simulated potentials are in volts.
"""

from importlib.metadata import PackageNotFoundError, version

from ttsim3d.config import DoseFilterConfig, PotentialConfig, Simulator, StructureConfig
from ttsim3d.filters import apply_fourier_filters, dose_filter_3d, mtf_filter_3d
from ttsim3d.io import write_mrc
from ttsim3d.potential import (
    auto_sublattice_radius,
    auto_upsampling,
    simulate_potential,
)
from ttsim3d.structure import adjust_b_factors, load_structure

try:
    __version__ = version("ttsim3d")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

__all__ = [
    "DoseFilterConfig",
    "PotentialConfig",
    "Simulator",
    "StructureConfig",
    "__version__",
    "adjust_b_factors",
    "apply_fourier_filters",
    "auto_sublattice_radius",
    "auto_upsampling",
    "dose_filter_3d",
    "load_structure",
    "mtf_filter_3d",
    "simulate_potential",
    "write_mrc",
]
