# ttsim3d

[![License](https://img.shields.io/pypi/l/ttsim3d.svg?color=green)](https://github.com/jdickerson95/ttsim3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ttsim3d.svg?color=green)](https://pypi.org/project/ttsim3d)
[![Python Version](https://img.shields.io/pypi/pyversions/ttsim3d.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/ttsim3d/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/ttsim3d)

Simulate 3D electrostatic potential maps from a PDB file in PyTorch.
This package currently replicates theory laid out in [Benjamin & Grigorieffa (2021)](https://doi.org/10.1107/S2052252521008538).

For a full list of changes, see the [CHANGELOG](CHANGELOG.md).

## Installation

`ttsim3d` is available on PyPi and can be installed via

```zsh
pip install ttsim3d
```

### From source
To create a source installation, first download/clone the repository, then run the install command
```zsh
git clone https://github.com/teamtomo/ttsim3d.git
cd ttsim3d
pip install -e .
```

Optional development and testing dependencies can also be installed by running
```zsh
pip install -e ".[dev,test]"
```

## Running CLI program

Installation of the package creates the executable program `ttsim3d-cli` which takes in a PDB file along with other simulation options and outputs the simulated 3D scattering potential to a .mrc file. 
All options for the program can be printed by running:
```zsh
ttsim3d-cli --help
```

The following are descriptions of each of the options for the program

| Argument | Type | Description
|----------|------|------------
| `--pixel-spacing`                                                     | float     | Value greater than zero for the size of each pixel/voxel, in units of Angstroms.
| `--volume-shape`                                                      | list      | List of integers specifying the output volume shape in pixels. For example, `'[256, 256, 256]'`.
| `--pdb-filepath`                                                      | path      | Path to the .pdb file to simulate
| `--b-factor-scaling`                                                  | float     | Multiplicative scaling to apply to atom b-factors in the .pdb file. Default is `1.0`.
| `--additional-b-factor`                                               | float     | Additional b-factor to add to all atoms in the .pdb file. Default is `0.0`.
| `--config-voltage`                                                    | float     | Microscope voltage, in keV, to use for the simulation.
| `--config-apply-dose-weighting` / `--no-config-apply-dose-weighting`  | selection | Option for choosing weather to apply cumulative dose weighting to the simulation. Default is `True`.
| `--config-crit-exposure-bfactor`                                      | float     | The critical exposure b-factor to use in the dose weighting. If `-1`, then use the resolution-dependent curve described in [Grant & Grigorieff (2015)](https://doi.org/10.7554/eLife.06980). Default is `-1`.
| `--config-dose-filter-modify-signal`                                  | str       | Either `"None"`, `"sqrt"`, or `"rel_diff"`. Default is `"None"`.
| `--config-dose-start`                                                 | float     | Beginning exposure for the cumulative dose filter in e-/A^2. Must be at least `0.0`. Default is `0.0`.
| `--config-dose-end`                                                   | float     | Ending exposure for the cumulative dose filter in e-/A^2. Must be at least `0.0`. Default is `30.0`.
| `--config-apply-dqe` / `--no-config-apply-dqe`                        | selection | Option for choosing weather to apply DQE filter to the final volume.
| `--config-mtf-reference`                                              | path      | Path to RELION-style .star file holding the modulation transfer function frequencies and amplitudes.
| `--config-upsampling`                                                 | int       | Integer specifying the level of upsampling used in the simulation. Default is `-1` which corresponds to automatic calculation.
| `--config-store-volume` / `--no-config-store-volume`                  | selection | Unused for CLI program.
| `--mrc-filepath`                                                      | path      | The file path to save the MRC file.
| `--gpu-ids`                                                           | list      | A list of GPU IDs to use for the simulation.
| `--help`                                                              | None      | Show this message and exit.

## Python objects

There are two user-facing classes in `ttsim3d` built upon Pydantic models for validating inputs and simulating a volume.
The first class, `ttsim3d.models.Simulator`, holds reference to a PDB file and basic simulation parameters related to that structure.
The second class, `ttsim3d.models.SimulatorConfig` is used to configure more advanced options, such as dose weighting and simulation upsampling.
An extremely basic use of these objects to run a simulation looks like
```python
from ttsim3d.models import Simulator, SimulatorConfig

# Instantiate the configuration object 
sim_conf = SimulatorConfig(
    voltage=300.0,  # in keV
    apply_dose_weighting=True,
    dose_start=0.0,  # in e-/A^2
    dose_end=35.0,   # in e-/A^2
    upsampling=-1,   # auto
)

# Instantiate the simulator
sim = Simulator(
    pdb_filepath="some/path/to/structure.pdb",
    pixel_spacing=1.25,  # Angstroms
    volume_shape=(256, 256, 256),
    b_factor_scaling=1.0,
    additional_b_factor=15.0,  # Add to all atoms
)

# Run the simulation
volume = sim.run()
print(type(volume))  # torch.Tensor
print(volume.shape)  # (256, 256, 256)

# OR export the simulation to a mrc file
mrc_filepath = "some/path/to/simulated_structure/mrc"
sim.export_to_mrc(mrc_filepath)
```
