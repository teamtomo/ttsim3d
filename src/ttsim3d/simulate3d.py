"""Simulation of 3D volume and associated helper functions."""

from typing import Literal, Union

import einops
import numpy as np
import torch
from torch_fourier_filter.ctf import calculate_relativistic_electron_wavelength
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid

from ttsim3d.device_handler import calculate_batches, get_device
from ttsim3d.grid_coords import (
    fourier_rescale_3d_force_size,
    get_atom_voxel_indices,
    get_upsampling,
    get_voxel_neighborhood_offsets,
)
from ttsim3d.scattering_potential import (
    get_scattering_potential_of_voxel_batch,
)

BOND_SCALING_FACTOR = 1.043
PIXEL_OFFSET = 0.5
MAX_SIZE = 1536
ALLOWED_DOSE_FILTER_MODIFICATIONS = ["None", "sqrt", "rel_diff"]


def _calculate_lead_term(beam_energy_kev: float, sim_pixel_spacing: float) -> float:
    """Calculate the lead term for the scattering potential.

    Parameters
    ----------
    beam_energy_kev : float
        The beam energy in kV.
    sim_pixel_spacing : float
        The pixel spacing for the final simulation in Angstroms.

    Returns
    -------
    float
        The lead term for the scattering potential.
    """
    beam_energy_ev = beam_energy_kev * 1000
    wavelength_m = calculate_relativistic_electron_wavelength(beam_energy_ev)
    wavelength_A = wavelength_m * 1e10
    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)

    return float(lead_term)


def _validate_dose_filter_inputs(
    dose_filter_modify_signal: str,
    dose_filter_critical_bfactor: float,
) -> None:
    """Ensure dose filter inputs are valid."""
    if dose_filter_modify_signal not in ALLOWED_DOSE_FILTER_MODIFICATIONS:
        raise ValueError(
            f"Invalid dose filter modification method: {dose_filter_modify_signal}. "
            f"Allowed methods are: {ALLOWED_DOSE_FILTER_MODIFICATIONS}"
        )

    if dose_filter_critical_bfactor == -1:
        return

    if dose_filter_critical_bfactor < 0:
        raise ValueError(
            "Critical B factor for dose filter must either be -1 (use critical "
            "exposure dose weighting) or a positive float. Given: "
            f"{dose_filter_critical_bfactor}"
        )


def _validate_dqe_filter_inputs(
    apply_dqe: bool,
    mtf_frequencies: torch.Tensor,
    mtf_amplitudes: torch.Tensor,
) -> None:
    """Ensure DQE filter inputs are valid."""
    if apply_dqe:
        if mtf_frequencies is None:
            raise ValueError(
                "If 'apply_dqe' is True, 'mtf_frequencies' must be provided. "
                "Got 'None'."
            )

        if mtf_amplitudes is None:
            raise ValueError(
                "If 'apply_dqe' is True, 'mtf_amplitudes' must be provided. "
                "Got 'None'."
            )

        if mtf_amplitudes.shape != mtf_frequencies.shape:
            raise ValueError(
                "The 'mtf_frequencies' and 'mtf_amplitudes' tensors must have "
                "the same shape. Got shapes: "
                f"{mtf_frequencies.shape} and {mtf_amplitudes.shape}."
            )


def _setup_sim3d_upsampling(
    sim_pixel_spacing: float,
    sim_volume_shape: tuple[int, int, int],
    upsampling: int,
) -> tuple[int, float, tuple[int, int, int]]:
    """Helper function to calculate upsampling factor and related values.

    Parameters
    ----------
    sim_pixel_spacing : float
        Desired final pixel spacing for simulation in Angstroms.
    sim_volume_shape : tuple[int, int, int]
        Desired final shape of the simulation volume. NOTE: simulations currently
        only support cubic volumes, but this is not explicitly checked.
    upsampling : int
        The upsampling factor as an int greater than 1 or -1. If -1, the
        upsampling factor is calculated automatically.

    Returns
    -------
    tuple[int, float, tuple[int, int, int]]
        The upsampling factor, the upsampled pixel size, and the upsampled shape.
    """
    if upsampling == -1:
        upsampling = get_upsampling(
            sim_pixel_spacing, sim_volume_shape[0], max_size=MAX_SIZE
        )
    elif upsampling < 1:
        raise ValueError("Upsampling factor must be greater than 1 (or -1 for auto)")

    upsampled_pixel_size = sim_pixel_spacing / upsampling
    upsampled_shape = tuple(np.array(sim_volume_shape) * upsampling)

    return upsampling, upsampled_pixel_size, upsampled_shape


def _setup_upsampling_coords(
    atom_positions_zyx: torch.Tensor,  # shape (N, 3)
    upsampled_pixel_size: float,
    upsampled_shape: tuple[int, int, int],
    mean_b_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to calculate the voxel indices and offsets for upsampling.

    Parameters
    ----------
    atom_positions_zyx : torch.Tensor
        The atom coordinates (continuous) in Angstroms. Shape (N, 3) where N is
        the number of atoms.
    upsampled_pixel_size : float
        The pixel size in Angstroms for the upsampled volume.
    upsampled_shape : tuple[int, int, int]
        The shape of the upsampled volume.
    mean_b_factor : float
        The mean B factor of the atoms. Used to estimate the neighborhood size
        necessary for simulation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The atom indices, atom dds, and voxel offsets for the neighborhood.
        Each tensor has shape (N, 3).
    """
    # Calculate the voxel coordinates of each atom
    atom_indices, atom_dds = get_atom_voxel_indices(
        atom_zyx=atom_positions_zyx,
        upsampled_pixel_size=upsampled_pixel_size,
        upsampled_shape=upsampled_shape,
        offset=PIXEL_OFFSET,
    )

    # Get the voxel offsets for the neighborhood around the atom voxel
    voxel_offsets_flat = get_voxel_neighborhood_offsets(
        mean_b_factor=mean_b_factor,
        upsampled_pixel_size=upsampled_pixel_size,
    )

    return atom_indices, atom_dds, voxel_offsets_flat


def setup_atomwise_scattering_potentials_simulation(
    atom_positions_zyx: torch.Tensor,  # shape (N, 3)
    atom_b_factors: torch.Tensor,
    sim_pixel_spacing: float,
    sim_volume_shape: tuple[int, int, int],
    upsampling: int = -1,
) -> dict:
    """Simulates the scattering potentials for each atom around its neighborhood.

    Parameters
    ----------
    atom_positions_zyx : torch.Tensor
        The atom coordinates in Angstroms. Shape (N, 3) where N is the number
        of atoms.
    atom_ids : list[str]
        The atom IDs as a list of uppercase element symbols.
    atom_b_factors : torch.Tensor
        The atom B factors.
    sim_pixel_spacing : float
        The pixel spacing for the final simulation in Angstroms.
    sim_volume_shape : tuple[int, int, int]
        The shape of the final simulation volume.
    lead_term : float
        The lead term for the scattering potential.
    upsampling : int
        The upsampling factor. If -1, the upsampling factor is calculated
        automatically.

    Returns
    -------
    dict
        A dictionary containing the atom indices, atom dds, voxel offsets,
        upsampled shape, upsampled pixel size, and actual upsampling factor.
    """
    actual_upsampling, upsampled_pixel_size, upsampled_shape = _setup_sim3d_upsampling(
        sim_pixel_spacing=sim_pixel_spacing,
        sim_volume_shape=sim_volume_shape,
        upsampling=upsampling,
    )

    mean_b_factor = torch.mean(atom_b_factors)
    atom_indices, atom_dds, voxel_offsets_flat = _setup_upsampling_coords(
        atom_positions_zyx=atom_positions_zyx,
        upsampled_pixel_size=upsampled_pixel_size,
        upsampled_shape=upsampled_shape,
        mean_b_factor=mean_b_factor,
    )

    return {
        "atom_indices": atom_indices,
        "atom_dds": atom_dds,
        "voxel_offsets_flat": voxel_offsets_flat,
        "upsampled_shape": upsampled_shape,
        "upsampled_pixel_size": upsampled_pixel_size,
        "actual_upsampling": actual_upsampling,
    }


def simulate_atomwise_scattering_potentials(
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    lead_term: float,
    atom_indices: torch.Tensor,
    atom_dds: torch.Tensor,
    voxel_offsets_flat: torch.Tensor,
    upsampled_pixel_size: float,
) -> torch.Tensor:
    """Simulates the scattering potentials for each atom around its neighborhood.

    Parameters
    ----------
    atom_ids : list[str]
        The atom IDs as a list of uppercase element symbols.
    atom_b_factors : torch.Tensor
        The atom B factors.
    lead_term : float
        The lead term for the scattering potential.
    atom_indices : torch.Tensor
        The atom indices.
    atom_dds : torch.Tensor
        The atom displacements from integer coordinates.
    voxel_offsets_flat : torch.Tensor
        The voxel offsets for the neighborhood around the atom voxel.
    upsampled_pixel_size : float
        The upsampled pixel size in Angstroms.

    Returnss
    -------
    dict
        A dictionary containing the neighborhood potentials and voxel positions.

    """
    # Reshaping tensors for future broadcasting
    atom_pos = einops.rearrange(atom_indices, "a d -> a 1 d")
    atom_dds = einops.rearrange(atom_dds, "a d -> a 1 d")
    voxel_offsets_flat = einops.rearrange(voxel_offsets_flat, "v d -> 1 v d")

    # Calculate voxel positions relative to atom center
    voxel_positions = atom_pos + voxel_offsets_flat  # (n_atoms, h*w*d, 3)

    # Calculate relative coordinates of each voxel in the neighborhood
    relative_coords = (
        voxel_positions - atom_pos - atom_dds - PIXEL_OFFSET
    ) * upsampled_pixel_size

    ########################################
    ### Scattering potential calculation ###
    ########################################

    neighborhood_potentials = get_scattering_potential_of_voxel_batch(
        zyx_coords1=relative_coords,  # shape(a, v, d)
        zyx_coords2=relative_coords + upsampled_pixel_size,  # shape(a, v, d)
        atom_ids=atom_ids,
        atom_b_factors=atom_b_factors,
        lead_term=lead_term,
    )

    return {
        "neighborhood_potentials": neighborhood_potentials,
        "voxel_positions": voxel_positions,
    }


def place_voxel_neighborhoods_in_volume(
    neighborhood_potentials: torch.Tensor,  # shape (N, h*w*d)
    voxel_positions: torch.LongTensor,  # shape (N, h*w*d, 3)
    final_volume: torch.Tensor,  # shape (H, W, D)
) -> torch.Tensor:
    """Places pre-calculate voxels of (N, h, w, d) into the volume (H, W, D).

    Parameters
    ----------
    neighborhood_potentials : torch.Tensor
        The pre-calculated scattering potentials in voxel neighborhoods around
        each atom. Shape (N, h*w*d).
    voxel_positions : torch.LongTensor
        The voxel offset positions for each of the neighborhoods. Shape (N, 3)
        with last dim representing (x, y z).
    final_volume : torch.Tensor
        The final volume to place the neighborhoods into. Shape (H, W, D).

    Returns
    -------
    torch.Tensor
        The final volume with the neighborhoods placed in.
    """
    index_positions = voxel_positions.long()
    try:
        final_volume.index_put_(
            indices=(
                index_positions[:, :, 0],
                index_positions[:, :, 1],
                index_positions[:, :, 2],
            ),
            values=neighborhood_potentials,
            accumulate=True,
        )
    except IndexError:
        raise ValueError("Error: Your box size is smaller than the potential") from None

    return final_volume


def calculate_simulation_dose_filter_3d(
    shape: tuple[int, int, int],
    dose_start: float,
    dose_end: float,
    critical_bfactor: float,
    modify_signal: Literal["None", "sqrt", "rel_diff"],
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Helper function to calculate a cumulative dose filter for a simulation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        The requested return shape of the dose filter.
    dose_start : float
        The starting dose for the dose filter in units of e-/A^2.
    dose_end : float
        The ending dose for the dose filter in units of e-/A^2.
    critical_bfactor : float
        The critical B factor for the dose filter. If -1, the Grant and
        Grigorieff (2015) critical exposure dose weighting is used.
    modify_signal : Literal["None", "sqrt", "rel_diff"]
        The method to modify the signal after applying the dose filter. Options
        are "None", "sqrt", and "rel_diff".
        - 'None': No modification is applied.'
        - 'sqrt': The square root of the filter is applied.
        - 'rel_diff': Filter becomes 1 - (1 - filter) / (1 + filter).
    rfft : bool
        If True, the filter is returned in rfft format. Default is True.
    fftshift : bool
        If True, the filter is fftshifted. Default is False.
    device : torch.device
        The device to use for the calculation. Default is None.

    Returns
    -------
    torch.Tensor
    """
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=shape,
        start_exposure=dose_start,
        end_exposure=dose_end,
        crit_exposure_bfactor=critical_bfactor,
        rfft=rfft,
        fftshift=fftshift,
    ).to(device)

    if modify_signal == "None":
        pass
    elif modify_signal == "sqrt":
        dose_filter = torch.sqrt(dose_filter)
    elif modify_signal == "rel_diff":
        denominator = 1 + dose_filter
        epsilon = 1e-10
        denominator = torch.clamp(denominator, min=epsilon)
        modification = 1 - (1 - dose_filter) / denominator
        dose_filter = modification

    return dose_filter


def apply_simulation_filters(
    upsampled_volume: torch.Tensor,
    actual_upsampling: int,
    final_shape: tuple[int, int, int],
    apply_dose_weighting: bool,
    dose_start: float,
    dose_end: float,
    dose_filter_modify_signal: Literal["None", "sqrt", "rel_diff"],
    dose_filter_critical_bfactor: float,
    apply_dqe: bool,
    mtf_frequencies: torch.Tensor,
    mtf_amplitudes: torch.Tensor,
) -> torch.Tensor:
    """Apply filters to the simulated volume.

    This function does the following:
    1. Apply dose weighting in the upsampled Fourier volume, if requested
    2. Fourier crop back to the desired size (final_shape)
    3. Apply DQE to the volume in Fourier space
    4. Inverse FFT to get the final real-space simulated volume

    Parameters
    ----------
    upsampled_volume : torch.Tensor
        The upsampled volume in Fourier space.
    actual_upsampling : int
        The actual upsampling factor used for the simulation.
    final_shape : tuple[int, int, int]
        The final shape of the simulated volume.
    apply_dose_weighting : bool
        If True, apply dose weighting to the simulation.
    dose_start : float
        The starting dose for the dose weighting filter in units of e-/A^2.
    dose_end : float
        The ending dose for the dose weighting filter in units of e-/A^2.
    dose_filter_modify_signal : Literal["None", "sqrt", "rel_diff"]
        The method to modify the signal after applying the dose filter. Options
        are "None", "sqrt", and "rel_diff".
    dose_filter_critical_bfactor : float
        The critical B factor for the dose filter. If -1, the Grant and
        Grigorieff (2015) critical exposure dose weighting is used.
    apply_dqe : bool
        If True, apply the detective quantum efficiency (DQE) filter to the
        simulation. Applied in regular (not upsampled) Fourier space.
    mtf_frequencies : torch.Tensor
        The frequencies for the modulation transfer function (MTF) filter in
        units of inverse pixels ranging from 0 to 0.5.
    mtf_amplitudes : torch.Tensor
        The amplitudes for the modulation transfer function (MTF) filter at the
        corresponding frequencies.

    Returns
    -------
    torch.Tensor
        The final simulated volume.

    """
    device = upsampled_volume.device

    upsampled_volume_rfft = torch.fft.rfftn(upsampled_volume, dim=(-3, -2, -1))
    upsampled_shape = upsampled_volume.shape

    # Calculate and apply dose filter, if requested
    if apply_dose_weighting:
        dose_filter = calculate_simulation_dose_filter_3d(
            shape=upsampled_shape,
            dose_start=dose_start,
            dose_end=dose_end,
            critical_bfactor=dose_filter_critical_bfactor,
            modify_signal=dose_filter_modify_signal,
            rfft=True,
            fftshift=False,
            device=device,
        )

        upsampled_volume_rfft *= dose_filter

    # Fourier crop back to desired size
    if actual_upsampling != 1:
        upsampled_volume_rfft = fourier_rescale_3d_force_size(
            volume_fft=upsampled_volume_rfft,
            volume_shape=upsampled_shape,
            target_size=final_shape[0],  # TODO: pass this arg as a tuple
            rfft=True,
            fftshift=False,
        )

    # Apply DQE, if requested
    if apply_dqe:
        mtf = make_mtf_grid(
            image_shape=final_shape,
            mtf_frequencies=mtf_frequencies,
            mtf_amplitudes=mtf_amplitudes,
            rfft=True,
            fftshift=False,
            device=device,
        ).to(device)
        upsampled_volume_rfft *= mtf

    # Inverse FFT
    cropped_volume = torch.fft.irfftn(
        upsampled_volume_rfft,
        s=final_shape,
        dim=(-3, -2, -1),
    )
    # NOTE: ifftshift not needed since volume here was never fftshifted
    # cropped_volume = torch.fft.ifftshift(cropped_volume, dim=(-3, -2, -1))

    return cropped_volume


def simulate3d(
    atom_positions_zyx: torch.Tensor,
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    beam_energy_kev: float,
    sim_pixel_spacing: float,
    sim_volume_shape: tuple[int, int, int],
    requested_upsampling: int = -1,
    apply_dose_weighting: bool = True,
    dose_start: float = 0.0,
    dose_end: float = 30.0,
    dose_filter_modify_signal: Literal["None", "sqrt", "rel_diff"] = "None",
    dose_filter_critical_bfactor: float = -1,
    apply_dqe: bool = False,
    mtf_frequencies: torch.Tensor = None,
    mtf_amplitudes: torch.Tensor = None,
    device: Union[int, str, list[int], list[str]] = "cuda:0",
    atom_batch_size: int = 16384,  # 2^14
) -> torch.Tensor:
    """Simulate 3D electron scattering volume with requested parameters.

    Parameters
    ----------
    atom_positions_zyx : torch.Tensor
        The atom positions in Angstroms. Shape (N, 3) where N is the number of
        atoms.
    atom_ids : list[str]
        The atomic IDs of each atom in the model.
    atom_b_factors : torch.Tensor
        The B factors of each atom in the model.
    beam_energy_kev : float
        The electron beam energy in kV.
    sim_pixel_spacing : float
        The pixel spacing for the final simulation in Angstroms.
    sim_volume_shape : tuple[int, int, int]
        The shape of the final simulation volume. Center of volume is at
        the origin (0.0, 0.0, 0.0).
    requested_upsampling : int
        The upsampling factor to use for the simulation. If -1, the upsampling
        factor is calculated automatically.
    apply_dose_weighting : bool
        If True, apply dose weighting to the simulation.
    dose_start : float
        The starting dose for the dose weighting filter in units of e-/A^2.
        Default is 0.0.
    dose_end : float
        The ending dose for the dose weighting filter in units of e-/A^2.
        Default is 30.0.
    dose_filter_modify_signal : str
        The method to modify the signal after applying the dose filter. Options
        are "None", "sqrt", and "rel_diff". Default is "None".
        - 'None': No modification is applied.
        - 'sqrt': The square root of the filter is applied.
        - 'rel_diff': Filter becomes 1 - (1 - filter) / (1 + filter).
    dose_filter_critical_bfactor : float
        The critical B factor for the dose filter. Default is -1, which uses
        the Grant and Grigorieff (2015) critical exposure dose weighting.
    apply_dqe : bool
        If True, apply the detective quantum efficiency (DQE) filter to the
        simulation. If True, both 'mtf_frequencies' and 'mtf_amplitudes' must
        be provided. Default is False.
    mtf_frequencies : torch.Tensor
        The frequencies for the modulation transfer function (MTF) filter in
        units of inverse pixels ranging from 0 to 0.5. Required if 'apply_dqe'
        is True.
    mtf_amplitudes : torch.Tensor
        The amplitudes for the modulation transfer function (MTF) filter at the
        corresponding frequencies. Must be the same length as
        'mtf_frequencies'. Required if 'apply_dqe' is True.
    device : Optional[Union[int, str, list[int], list[str]]]
        The device to run the simulation on. Default is "cuda:0", but other possible
        values include "cpu" for CPU execution or integer "0" which will use the
        first available CUDA device. NOTE: If a list of devices is provided,
        then only the first device in the list will be used.
    atom_batch_size : int
        The number of atoms to process (simulate the scattering potentials of) at a
        single time. This is partially controls the memory usage. If -1, the batch size
        calculated automatically. Default is 16384 (2^14).

    Returns
    -------
    torch.Tensor
        The simulated 3D volume in real space.
    """
    # Get compute device
    device = get_device(device)

    # Move input tensors to device
    atom_positions_zyx = atom_positions_zyx.to(device)
    atom_b_factors = atom_b_factors.to(device)
    if mtf_frequencies is not None:
        mtf_frequencies = mtf_frequencies.to(device)
    if mtf_amplitudes is not None:
        mtf_amplitudes = mtf_amplitudes.to(device)

    # Validate portions of the input before continuing
    _validate_dose_filter_inputs(
        dose_filter_modify_signal, dose_filter_critical_bfactor
    )
    _validate_dqe_filter_inputs(apply_dqe, mtf_frequencies, mtf_amplitudes)

    # Calculate the atom-wise scattering potentials
    lead_term = _calculate_lead_term(beam_energy_kev, sim_pixel_spacing)

    # set up atomwise scattering potentials
    setup_results = setup_atomwise_scattering_potentials_simulation(
        atom_positions_zyx=atom_positions_zyx,
        atom_b_factors=atom_b_factors,
        sim_pixel_spacing=sim_pixel_spacing,
        sim_volume_shape=sim_volume_shape,
        upsampling=requested_upsampling,
    )

    upsampled_shape = setup_results["upsampled_shape"]
    actual_upsampling = setup_results["actual_upsampling"]

    # Nowsplit to batches
    upsampled_volume = torch.zeros(upsampled_shape, dtype=torch.float32, device=device)

    # Only do automatic batch calculation if atom_batch_size is -1
    if atom_batch_size == -1:
        num_batches, atoms_per_batch = calculate_batches(
            setup_results, upsampled_volume
        )
    else:
        atoms_per_batch = atom_batch_size
        num_batches = atom_positions_zyx.shape[0] // atoms_per_batch + 1

    for batch_idx in range(num_batches):
        start_idx = batch_idx * atoms_per_batch
        end_idx = min(
            (batch_idx + 1) * atoms_per_batch, setup_results["atom_indices"].shape[0]
        )

        # Process a batch of atoms
        scattering_results = simulate_atomwise_scattering_potentials(
            atom_ids=atom_ids[start_idx:end_idx],
            atom_b_factors=atom_b_factors[start_idx:end_idx],
            lead_term=lead_term,
            atom_indices=setup_results["atom_indices"][start_idx:end_idx],
            atom_dds=setup_results["atom_dds"][start_idx:end_idx],
            voxel_offsets_flat=setup_results["voxel_offsets_flat"],
            upsampled_pixel_size=setup_results["upsampled_pixel_size"],
        )

        neighborhood_potentials = scattering_results["neighborhood_potentials"]
        voxel_positions = scattering_results["voxel_positions"]

        # Update the upsampled volume with the current batch
        upsampled_volume = place_voxel_neighborhoods_in_volume(
            neighborhood_potentials=neighborhood_potentials,
            voxel_positions=voxel_positions,
            final_volume=upsampled_volume,
        )

    # Apply filters to the simulation
    final_volume = apply_simulation_filters(
        upsampled_volume=upsampled_volume,
        actual_upsampling=actual_upsampling,
        final_shape=sim_volume_shape,
        apply_dose_weighting=apply_dose_weighting,
        dose_start=dose_start,
        dose_end=dose_end,
        dose_filter_modify_signal=dose_filter_modify_signal,
        dose_filter_critical_bfactor=dose_filter_critical_bfactor,
        apply_dqe=apply_dqe,
        mtf_frequencies=mtf_frequencies,
        mtf_amplitudes=mtf_amplitudes,
    )

    return final_volume
