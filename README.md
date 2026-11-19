# ttsim3d

[![License](https://img.shields.io/pypi/l/ttsim3d.svg?color=green)](https://github.com/jdickerson95/ttsim3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ttsim3d.svg?color=green)](https://pypi.org/project/ttsim3d)
[![Python Version](https://img.shields.io/pypi/pyversions/ttsim3d.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/ttsim3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/ttsim3d/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/ttsim3d)

Simulate 3D electrostatic potential maps from atomic structures (PDB/mmCIF) in PyTorch.

`ttsim3d` is a thin layer over the [teamtomo](https://github.com/teamtomo/teamtomo) primitives:

| Step | Package |
|---|---|
| Read structure, bonding annotation | `mmdf`, `torch-structure-manipulation` |
| Potential under the isolated-atom superposition approximation (Peng 1996 elemental or bonded scattering factors, averaged over each voxel) | `torch-calculate-electrostatic-potential` |
| Upsampled simulation -> Fourier crop | `torch-fourier-rescale` |
| Cumulative dose filter, detector MTF | `torch-fourier-filter` |

For a full list of changes, see the [CHANGELOG](CHANGELOG.md).

## Installation

```zsh
pip install ttsim3d
```

From source, including development and test dependencies:

```zsh
git clone https://github.com/jdickerson95/ttsim3d.git
cd ttsim3d
pip install -e ".[dev,test]"
```

## Units and conventions

- Simulated potentials are in **volts**.
- Coordinates and pixel spacings are in Angstroms. Arrays are ordered ZYX.
- The physical origin is at voxel index `N // 2` along every axis (the FFT
  convention used by RELION and cisTEM). By default the structure is centered
  on its mean atom position; pass `center=False` to keep the coordinates of a
  pre-aligned model.
- Atomic B-factors are used as given (Debye–Waller factor `exp(-B s^2)`), and
  atoms are weighted by their occupancy.

## Python API

### Configured simulations: `Simulator`

`Simulator` is a serializable (Pydantic) description of a simulation:

```python
from ttsim3d import Simulator

sim = Simulator(
    structure={"path": "some/path/to/structure.pdb", "b_factor_offset": 15.0},
    pixel_spacing=1.25,           # Angstroms
    volume_shape=(256, 256, 256),
    dose_filter={"end": 35.0},    # e-/A^2; None disables the dose filter
    mtf="k2_300kV",               # bundled MTF name or star file; None (default) disables it
)

volume = sim.run(device="cuda:0")         # filtered volume, torch.Tensor (256, 256, 256)
potential = sim.simulate_potential()      # unfiltered potential in volts
sim.export_to_mrc("simulated_structure.mrc", device="cuda:0")
```

The model holds only settings, so they can be changed between runs (e.g.
`sim.pixel_spacing = 1.5`). Settings round-trip through YAML or JSON:

```python
sim.to_yaml("simulation_config.yaml")
sim = Simulator.from_yaml("simulation_config.yaml")
```

```yaml
dose_filter:
  critical_bfactor: null   # null: Grant & Grigorieff (2015) critical exposure
  end: 35.0
  modify_signal: none      # none | sqrt | rel_diff
  start: 0.0
mtf: k2_300kv
pixel_spacing: 1.25
potential:
  batch_size: 4096
  bonded_fallback: elemental
  scattering_factors: peng_elemental   # or peng_bonded
  sublattice_radius: null              # null: chosen from the B-factors
  upsampling: null                     # null: chosen from the pixel spacing
structure:
  b_factor_offset: 15.0
  b_factor_scale: 1.0
  center: true
  path: some/path/to/structure.pdb
  remove_hydrogens: true
volume_shape:
- 256
- 256
- 256
```

### Building blocks

Each step is also available as a function:

```python
from ttsim3d import (
    adjust_b_factors,
    apply_fourier_filters,
    dose_filter_3d,
    load_structure,
    mtf_filter_3d,
    simulate_potential,
)

structure = load_structure("structure.cif", device="cuda:0")   # AtomicStructure
structure = adjust_b_factors(structure, scale=1.0, offset=15.0)

potential = simulate_potential(structure, 1.25, (256, 256, 256))  # volts

volume = apply_fourier_filters(
    potential,
    dose_filter_3d((256, 256, 256), 1.25, dose_end=35.0, device="cuda:0"),
    mtf_filter_3d((256, 256, 256), "k2_300kv", device="cuda:0"),
)
```

`simulate_potential` accepts any `torch_structure_manipulation.AtomicStructure`,
including batched positions (e.g. several poses of the same structure).

### Simulation accuracy options

- `upsampling`: the potential is evaluated on a grid `upsampling` times finer
  and Fourier cropped to the output size to reduce aliasing. By default
  (`None`) it is 4 above 1.5 Å/pixel, 2 above 0.75 Å/pixel, and 1 otherwise,
  limited so that the upsampled box stays below 1536 voxels.
- `sublattice_radius`: radius (Å) of the neighbourhood evaluated around each
  atom. By default it is chosen so the widest Gaussian term present falls to
  1e-3 of its peak (see `ttsim3d.auto_sublattice_radius`).
- `scattering_factors`: `"peng_elemental"` (default) or `"peng_bonded"`
  (bonding-environment-dependent factors of Shtyrov et al. 2026; the structure
  is annotated automatically when using `Simulator`).

### Detector MTFs

The detector MTF curves are bundled with `torch-fourier-filter`:

```python
from torch_fourier_filter.mtf import list_mtf_references

print(list_mtf_references())
```

Names are case-insensitive. A path to any RELION-style MTF star file also works.

## Command line

Installing the package provides `ttsim3d-cli`:

```zsh
ttsim3d-cli --structure structure.pdb --pixel-spacing 1.0 --volume-shape 256 256 256 \
    --mtf k3_standard_300kv_fl2 --device cuda:0 -o simulated.mrc

# or start from a YAML configuration and override some settings
ttsim3d-cli --config simulation_config.yaml --pixel-spacing 1.5 -o simulated.mrc
```

Run `ttsim3d-cli --help` for all options.

## Migrating from ttsim3d 0.4

Version 0.5 is a breaking release. See the [CHANGELOG](CHANGELOG.md) for the
reasons why simulated volumes differ.

| 0.4 | 0.5 |
|---|---|
| `from ttsim3d.models import Simulator, SimulatorConfig` | `from ttsim3d import Simulator` |
| `pdb_filepath`, `center_atoms`, `remove_hydrogens` | `structure.path`, `structure.center`, `structure.remove_hydrogens` |
| `b_factor_scaling`, `additional_b_factor` | `structure.b_factor_scale`, `structure.b_factor_offset` (B is no longer multiplied by 0.25) |
| `simulator_config.voltage` | removed (the potential in volts does not depend on it) |
| `simulator_config.upsampling = -1` | `potential.upsampling = None` |
| `simulator_config.atom_batch_size` | `potential.batch_size` |
| `apply_dose_weighting`, `dose_start`, `dose_end` | `dose_filter = {start, end}` or `None` |
| `crit_exposure_bfactor = -1` | `dose_filter.critical_bfactor = None` |
| `dose_filter_modify_signal = "None"` | `dose_filter.modify_signal = "none"` |
| `apply_dqe`, `mtf_reference` | `mtf = <name or path>` or `None` |
| `store_volume`, `Simulator.volume` | removed; use the return value of `run()` |
| `run(device=0)` or `run(device=[0, 1])` | `run(device="cuda:0")` |
| `ttsim3d.simulate3d.simulate3d(...)` | `ttsim3d.simulate_potential(...)` + `ttsim3d.apply_fourier_filters(...)` |
| `ttsim3d-cli --pdb-filepath ... --mrc-filepath ...` | `ttsim3d-cli --structure ... -o ...` |
