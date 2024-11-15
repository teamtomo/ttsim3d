"""Simulate a 3D electrostatic potential map from a PDB in pyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttsim3d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
