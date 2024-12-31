# ttsim3d

[![License](https://img.shields.io/pypi/l/ttsim3d.svg?color=green)](https://github.com/jdickerson95/ttsim3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ttsim3d.svg?color=green)](https://pypi.org/project/ttsim3d)
[![Python Version](https://img.shields.io/pypi/pyversions/ttsim3d.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/ttsim3d/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/ttsim3d)

Simulate a 3D electrostatic potential map from a PDB in pyTorch.

## Installation

Install via source using
```zsh
pip install -e .
```

To install the optional development and testing dependencies, use
```zsh
pip install -e ".[dev,test]"
```

## Running CLI program

Installation of the package creates the executable program `ttsim3d-cli` which takes a PDB file along with other simulation options as input and outputs the simulated 3D scattering potential to a .mrc file. 
All options for the program can be printed by running:
```zsh
ttsim3d-cli --help
```

## Python objects

TODO
