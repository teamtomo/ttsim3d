# ttsim3d

[![License](https://img.shields.io/pypi/l/ttsim3d.svg?color=green)](https://github.com/jdickerson95/ttsim3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ttsim3d.svg?color=green)](https://pypi.org/project/ttsim3d)
[![Python Version](https://img.shields.io/pypi/pyversions/ttsim3d.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/ttsim3d/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/ttsim3d)

Simulate 3D electrostatic potential maps from a PDB file in PyTorch.
This package currently replicates theory laid out in [Himes & Grigorieff (2021)](https://doi.org/10.1107/S2052252521008538).

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


| Option                        | Type                                  | Default       | Description                                                                                                                                                       |
| ------------------------------|---------------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--pdb-filepath`              | Path                                  | required      | The path to the PDB file containing the atomic structure to simulate.
| `--mrc-filepath`              | Path                                  | required      | File path to save simulated volume.
| `--pixel-spacing`             | float                                 | required      | The pixel spacing of the simulated volume in units of Angstroms. Must be greater than 0.
| `--volume-shape`              | (int, int, int)                       | required      | The shape of the simulated volume in pixels.
| `--voltage`                   | float                                 | `300.0`       | The voltage of the microscope in kV. Default is 300 kV.
| `--upsampling`                | int                                   | `-1`          | The upsampling factor to apply to the simulation. The default is -1 and corresponds to automatic calculation of the upsampling factor.
| `--b-factor-scaling`          | float                                 | `1.0`         | The scaling factor to apply to the B-factors of the atoms in the pdb file. The default is 1.0.
| `--additional-b-factor`       | float                                 | `0.0`         | Additional B-factor to apply to the atoms in the pdb file. The default is 0.0.
| `--apply-dose-weighting`      | bool                                  | `True`        | If True, apply dose weighting to the simulation. Default is True.
| `--crit-exposure-bfactor`     | float                                 | `-1.0`        | B-factor to use in critical exposure calculations. The default is -1 and corresponds to the fitted critical exposure function in Grant and Grigorieff, 2015.
| `--dose-filter-modify-signal` | Literal["None", "sqrt", "rel_diff"]   | `"None"`      | Signal modification to apply to the dose filter. Currently supports 'None', 'sqrt', and 'rel_diff'.
| `--dose-start`                | float                                 | `0.0`         | The starting dose in e/A^2.
| `--dose-end`                  | float                                 | `30.0`        | The ending dose in e/A^2.
| `--apply-dqe`                 | bool                                  | `True`        | If True, apply a DQE filter to the simulation.
| `--mtf-reference`             | Path or str                           | `"k2_300kV"`  | Path to the modulation transfer function (MTF) reference star file, or one of the known MTF reference files. Default is 'k2_300kV'.
| `--gpu-ids`                   | list[int]                             | unused        | A list of GPU IDs to use for the simulation. Currently unused.

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
    simulator_config=sim_conf,
)

# Run the simulation
volume = sim.run()
print(type(volume))  # torch.Tensor
print(volume.shape)  # (256, 256, 256)

# OR export the simulation to a mrc file
mrc_filepath = "some/path/to/simulated_structure/mrc"
sim.export_to_mrc(mrc_filepath)
```

### Working with configuration files

Simulation configurations can be saved to disk as either a YAML or JSON file by using the `to_yaml` or `to_json` methods of the `Simulator` class, respectively.
Assuming the same `sim` object defined as above, you can export the configuration to a YAML file like this:
```python
sim.to_yaml("some/path/to/simulation_config.yaml")
```

The contents of the YAML file will look something like this:
```yaml
additional_b_factor: 15.0
b_factor_scaling: 1.0
center_atoms: true
pdb_filepath: some/path/to/structure.pdb
pixel_spacing: 1.25
remove_hydrogens: true
volume_shape:
- 256
- 256
- 256
simulator_config:
  apply_dose_weighting: true
  apply_dqe: true
  atom_batch_size: 16384
  crit_exposure_bfactor: -1
  dose_end: 35.0
  dose_filter_modify_signal: None
  dose_start: 0.0
  mtf_reference: k2_300kv
  store_volume: true
  upsampling: -1
  voltage: 300.0
```

Similarly, you can load in a configuration from a YAML file by using the `from_yaml` class method of the `Simulator` class:
```python
from ttsim3d.models import Simulator

sim = Simulator.from_yaml("some/path/to/simulation_config.yaml")
```