# Changelog

## [unreleased] -- 14 August 2025
### Added
- `metadata_handler.py`: new module housing `make_metadata_file` function.
- `make_metadata_file`: utility to generate metadata with simulated volumes

## [v0.3.0] -- 17 June 2025

### Modified

- Renamed parameter `gpu_ids` in function `simulate3d` to `device`
- Device selection now defaults to `"cpu"` with GPUs selectable via integer or string (e.g. `0` or `"cuda:2"`)
- Connect CLI device selection into the program
- Connect atom batch size parameter in CLI to simulation program

### Fixed

- Fixes bug where pixel size was not being passed down to function simulating the dose weighting filter

## [v0.2.1] -- 19 January 2025

### Added

- `Simulator` and `SimulatorConfig` pydantic classes for parsing inputs and running simulations.
- Easy to use command line program `ttsim3d-cli` for basic simulations.
- Basic documentation and unit tests across the repository.

### Modified

- `ttsim3d.simulate3d.simulate3d` accepts different arguments than before.

### Fixed

- Minor bug fixes related to parsing input files.