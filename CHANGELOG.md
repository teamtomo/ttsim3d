# Changelog

## [Unreleased]

### Added

- Subtraction of the solvent a molecule displaces: the map becomes `A - rho * E`, with `E` the solvent-excluded volume of the model scaled to the mean of the unfiltered potential `A` inside it, before the dose filter and MTF. Set by the new `SimulatorConfig` fields `subtract_solvent` (default `False`) and `solvent_density` (`rho`, default 0.6 for water), `simulate3d(..., solvent_density=...)`, and `ttsim3d-cli --subtract-solvent --solvent-density`. New module `ttsim3d.solvent`.
- `E`, scaled and filtered as it enters the map, can be returned or written as its own map (`return_excluded_volume=True` in `simulate3d` and `Simulator.run`, `Simulator.export_to_mrc(excluded_volume_filepath=...)`, `ttsim3d-cli --excluded-volume-mrc`), so the map for any `rho` can be made later.

## [v0.4.0] -- 17 June 2025

### Modified

- Renamed parameter `gpu_ids` in function `simulate3d` to `device`
- Device selection now defaults to `"cpu"` with GPUs selectable via integer or string (e.g. `0` or `"cuda:2"`)
- Connect CLI device selection into the program
- Connect atom batch size parameter in CLI to simulation program

### Fixed

- Fixes bug where pixel size was not being passed down to function simulating the dose weighting filter

## [v0.3.0-v0.3.2] -- 16 May 2025

## [v0.2.1] -- 19 January 2025

### Added

- `Simulator` and `SimulatorConfig` pydantic classes for parsing inputs and running simulations.
- Easy to use command line program `ttsim3d-cli` for basic simulations.
- Basic documentation and unit tests across the repository.

### Modified

- `ttsim3d.simulate3d.simulate3d` accepts different arguments than before.

### Fixed

- Minor bug fixes related to parsing input files.