# Changelog

## [v0.5.0] -- Unreleased

Breaking release. The simulation is now built on the teamtomo v0.6 primitives
(`torch-calculate-electrostatic-potential`, `torch-structure-manipulation`,
`torch-fourier-rescale`, `torch-fourier-filter`). ttsim3d's own scattering
potential, grid, batching, and PDB code was removed. See the
[README](README.md#migrating-from-ttsim3d-04) for the old -> new API mapping.

### Simulated volumes change

- **Units are volts.** The cisTEM lead term `1.043*lambda/(8*px^2)` is gone, including
  its empirical bond scaling factor 1.043. Old values are approximately new values times
  factor `1.043*lambda*px / 47.8776` (e.g. ~4.3e-4 at 300 kV and 1 Å). This is a pure
  scale factor: with the old B-factor and neighborhood conventions reproduced, old
  and new volumes have a correlation of 0.99994.
- **B-factors are no longer multiplied by 0.25.** They enter as the standard
  Debye–Waller factor, so volumes are blurrier for the same model B-factors. Use
  `structure.b_factor_scale = 0.25` to approximate the old sharpness.
- **Larger per-atom neighbourhood.** The cisTEM-sized neighborhood
  (about 1–2 Å) truncated the Gaussian tails. The radius now follows the widest
  Gaussian present (about 4–5 Å for typical B-factors), which costs roughly
  2–3× more compute (a full 60S subunit at 0.95 Å, 400³, takes 16 s instead of 6 s on
  CPU).
- **Occupancies weight atoms.**
- **Scattering factors for Fr–U are correct.** The previous table had
  column-shifted entries for the heavy actinides.

### Fixed

- Upsampling factors other than 2 produced corrupted volumes: the Fourier crop
  shifted the half-length rfft axis, which only happens to work for factor 2.
  Automatic upsampling chooses 4 above 1.5 Å/pixel, so those simulations were
  affected (correlation with an un-upsampled simulation approx. 0.01).
- Odd volume sizes are centered at voxel `N // 2` with any upsampling factor.
- The CLI default MTF (`k2_300kV`) did not match a registry key, and the CLI and
  `SimulatorConfig` disagreed on whether the DQE filter is applied by default.
- The trailing empty atom batch when the atom count was a multiple of the batch
  size.
- The CLI no longer installs a top-level `programs` package.

### Added

- `scattering_factors="peng_bonded"` (bonding-environment-dependent scattering
  factors of Shtyrov et al. 2026) and `bonded_fallback`.
- `sublattice_radius` option and `auto_sublattice_radius`.
- A functional API: `load_structure`, `adjust_b_factors`, `simulate_potential`,
  `dose_filter_3d`, `mtf_filter_3d`, `apply_fourier_filters`, `write_mrc`.
- `Simulator.simulate_potential()` returns the unfiltered potential in volts,
  suitable as input to `torch-scattering`.
- Batched structures (several poses of one model) in `simulate_potential`.
- `ttsim3d-cli --config file.yaml`, with command line options overriding it.

### Modified

- New `Simulator` schema with `structure`, `potential`, `dose_filter`, and `mtf`
  sections; `SimulatorConfig` was removed. Sentinel values were replaced by
  `None` (`upsampling`, `critical_bfactor`); `dose_filter = None` and
  `mtf = None` disable the filters.
- `voltage` was removed: neither the potential in volts nor the 3D dose filter
  depends on it.
- `Simulator.run(atom_indices=...)` was removed; subset the `AtomicStructure`
  and call `simulate_potential` instead.
- `device` accepts a single device (`"cpu"`, `"cuda:0"`, or a `torch.device`).
  Integer and list forms and the host-memory-based automatic batching were
  removed.
- Detector MTF files moved to `torch-fourier-filter` (names are now lowercase
  file names, e.g. `k3_standard_300kv_fl2`, and case-insensitive).
- CLI options were renamed: `--structure`, `-o/--output`, `--mtf`,
  `--dose-filter/--no-dose-filter`, and so on.
- Requires Python >= 3.11.

### Known issues

- The dose filter from `torch-fourier-filter` is 0 at zero frequency, so
  dose-filtered volumes have zero mean, as in previous releases.

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