"""The main simulation function."""

import time

# from collections import namedtuple
from typing import Optional

import einops
import numpy as np
import torch
from torch_fourier_filter.ctf import calculate_relativistic_electron_wavelength
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid, read_mtf

from ttsim3d.device_handler import select_gpu
from ttsim3d.grid_coords import (
    fourier_rescale_3d_force_size,
    get_atom_voxel_indices,
    get_upsampling,
    get_voxel_neighborhood_offsets,
)
from ttsim3d.mrc_handler import tensor_to_mrc
from ttsim3d.pdb_handler import load_model, remove_hydrogens
from ttsim3d.scattering_potential import (
    get_scattering_potential_of_voxel_batch,
)

BOND_SCALING_FACTOR = 1.043
PIXEL_OFFSET = 0.5
MAX_SIZE = 1536


def _calculate_lead_term(beam_energy_kev: float, sim_pixel_spacing: float) -> float:
    """Calculate the lead term for the scattering potential."""
    beam_energy_ev = beam_energy_kev * 1000
    wavelength_m = calculate_relativistic_electron_wavelength(beam_energy_ev)
    wavelength_A = wavelength_m * 1e10
    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)

    return float(lead_term)


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
        The atom coordinates in Angstroms. Shape (N, 3) where N is the number
        of atoms.
    upsampled_pixel_size : float
        The pixel size in Angstroms for the upsampled volume.
    upsampled_shape : tuple[int, int, int]
        The shape of the upsampled volume.
    mean_b_factor : float
        The mean B factor of the atoms.

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


def simulate_atomwise_scattering_potentials(
    atom_positions_zyx: torch.Tensor,  # shape (N, 3)
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    sim_pixel_spacing: float,
    sim_volume_shape: tuple[int, int, int],
    lead_term: float,
    upsampling: int = -1,
) -> torch.Tensor:
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


    """
    upsampling, upsampled_pixel_size, upsampled_shape = _setup_sim3d_upsampling(
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
    coords1 = relative_coords
    coords2 = relative_coords + upsampled_pixel_size

    ########################################
    ### Scattering potential calculation ###
    ########################################

    neighborhood_potentials = get_scattering_potential_of_voxel_batch(
        zyx_coords1=coords1,  # shape(a, v, d)
        zyx_coords2=coords2,  # shape(a, v, d)
        atom_ids=atom_ids,
        atom_b_factors=atom_b_factors,
        lead_term=lead_term,
    )

    return {
        "neighborhood_potentials": neighborhood_potentials,
        "voxel_positions": voxel_positions,
        "upsampled_shape": upsampled_shape,
        "upsampled_pixel_size": upsampled_pixel_size,
        "upsampling": upsampling,
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
    final_volume.index_put_(
        indices=(
            index_positions[:, :, 0],
            index_positions[:, :, 1],
            index_positions[:, :, 2],
        ),
        values=neighborhood_potentials,
        accumulate=True,
    )

    return final_volume


def apply_simulation_filters(
    upsampled_volume_rfft: torch.Tensor,
    upsampled_shape: tuple[int, int, int],
    final_shape: tuple[int, int, int],
    upsampled_pixel_size: float,
    upsampling: int,
    dose_weighting: bool,
    num_frames: int,
    fluence_per_frame: float,
    dose_B: float,
    modify_signal: int,
    apply_dqe: bool,
    mtf_filename: str,
) -> torch.Tensor:
    """Apply filtering to to simulated volume.

    This function does the following:
    1. Apply dose weighting in the upsampled Fourier volume
    2. Fourier crop back to the desired size (final_shape)
    3. Apply DQE to the volume in Fourier space

    Parameters
    ----------
    upsampled_volume_rfft : torch.Tensor
        RFFT of the upsampled volume.
    upsampled_shape : tuple[int, int, int]
        The shape of the full upsampled volume.
    final_shape : tuple[int, int, int]
        The desired final shape of the final simulation volume.
    upsampled_pixel_size : float
        The pixel/voxel size of the upsampled volume.
    upsampling : int
        The upsampling factor.
    dose_weighting : bool
        If true, apply cumulative dose weighting.
    num_frames : int
        The number of frames for the simulation. Used for calculating the dose
        weighting filter
    fluence_per_frame : float
        The fluence per frame in units of (e-/A^2). Used for calculating the
        dose weighting filter.
    dose_B : float
        Parameter to choose the dose weighting filter. If -1, use the Grant
        Grigorieff critical exposure dose weighting.
    modify_signal : int
        Integer to determine how to modify the signal by the dose filter.
        Follows cisTEM convention of:
            1: 1 - (1 - filter) / (1 + filter)
            2: (filter)**0.5
            3: (filter)
    apply_dqe : bool
        If true, apply DQE to the volume. Requires an MTF file.
    mtf_filename : str
        The filename of the MTF file. Required if apply_dqe is True.

    Returns
    -------
    torch.Tensor
        The final simulated volume.

    """
    ####################################################################
    ### Filters to apply before Fourier cropping (on upsampled grid) ###
    ####################################################################

    # Dose weight
    if dose_weighting:
        print("Dose weighting")
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=upsampled_shape,
            num_frames=num_frames,
            start_exposure=0,
            pixel_size=upsampled_pixel_size,
            flux=fluence_per_frame,
            Bfac=dose_B,
            rfft=True,
            fftshift=False,
        )

        if modify_signal == 1:
            # Add small epsilon to prevent division by zero
            denominator = 1 + dose_filter
            epsilon = 1e-10
            denominator = torch.clamp(denominator, min=epsilon)
            modification = 1 - (1 - dose_filter) / denominator

            # Check for invalid values
            if torch.any(torch.isnan(modification)):
                print("Warning: NaN values in modification factor")
                modification = torch.nan_to_num(modification, nan=1.0)
            dose_filter = modification
        elif modify_signal == 2:
            dose_filter = dose_filter**0.5
        else:
            dose_filter = dose_filter

        upsampled_volume_rfft *= dose_filter

    # fourier crop back to desired output size
    if upsampling > 1:
        print("Fourier cropping back to desired size")
        upsampled_volume_rfft = fourier_rescale_3d_force_size(
            volume_fft=upsampled_volume_rfft,
            volume_shape=upsampled_shape,
            target_size=final_shape[0],  # TODO: pass this arg as a tuple
            rfft=True,
            fftshift=False,
        )

    #################################################################
    ### Filters to apply after Fourier cropping (on desired grid) ###
    #################################################################

    # If I apply dqe before Fourier cropping like cisTEM the output is the same.
    # I apply it after Fourier cropping
    if apply_dqe:
        print("Applying a DQE")
        mtf_frequencies, mtf_amplitudes = read_mtf(file_path=mtf_filename)
        mtf = make_mtf_grid(
            image_shape=final_shape,
            mtf_frequencies=mtf_frequencies,  # 1D tensor
            mtf_amplitudes=mtf_amplitudes,  # 1D tensor
            rfft=True,
            fftshift=False,
        )
        upsampled_volume_rfft *= mtf

    # inverse FFT
    cropped_volume = torch.fft.irfftn(
        upsampled_volume_rfft,
        s=(final_shape[0], final_shape[0], final_shape[0]),
        dim=(-3, -2, -1),
    )
    cropped_volume = torch.fft.ifftshift(cropped_volume, dim=(-3, -2, -1))

    return cropped_volume


def simulate3d(
    pdb_filename: str,
    output_filename: str,
    sim_volume_shape: tuple[int, int, int],
    sim_pixel_spacing: float,
    num_frames: int,
    fluence_per_frame: float,
    beam_energy_kev: float = 300,
    dose_weighting: bool = True,
    dose_B: float = -1,  # -1 is use Grant Grigorieff dose weighting
    apply_dqe: bool = True,
    mtf_filename: str = "",
    b_scaling: float = 1.0,
    added_B: float = 0.0,
    upsampling: int = -1,  # -1 is calculate automatically
    gpu_id: Optional[int] = -999,  # -999 cpu, -1 auto, 0 = gpuid
    modify_signal: int = 1,
) -> None:
    """
    Run the 3D simulation.

    Args:
        pdb_filename: The filename of the PDB file to use as the reference
            structure.
        output_filename: The filename of the output MRC file.
        sim_volume_shape: The shape of the simulation volume in voxels.
        sim_pixel_spacing: The pixel spacing of the simulation volume, in units
            of Angstroms.
        num_frames: The number of frames for the simulation. Used for dose
            weighting.
        fluence_per_frame: The fluence per frame, in units of (e-/A^2). Used
            for dose weighting.
        beam_energy_kev: The beam energy in keV.
        dose_weighting: Whether to apply dose weighting.
        dose_B: The B factor for dose weighting.
        apply_dqe: Whether to apply DQE.
        mtf_filename: The filename of the MTF file.
        b_scaling: The B scaling factor.
        added_B: The added B factor.
        upsampling: The upsampling factor.
        gpu_ids: The specified GPU id (-999 cpu, -1 auto)
        modify_signal: The signal modification factor.

    Returns
    -------
        None
    """
    # Select devices
    if gpu_id == -999:  # Special case for CPU-only
        device = torch.device("cpu")
    else:
        device = select_gpu(gpu_id)
    print(f"Using device: {device!s}")  # TODO: Move to logging

    # This is the main program
    start_time = time.time()

    #################
    ### Constants ###
    #################

    lead_term = _calculate_lead_term(beam_energy_kev, sim_pixel_spacing)

    #########################################
    ### PDB model to coords/ids/b-factors ###
    #########################################

    # Then load pdb (a separate file) and get non-H atom
    # list with zyx coords and isotropic b factors
    print("Loading PDB model")  # TODO: Move to logging
    atom_positions_zyx, atom_ids, atom_b_factors = load_model(pdb_filename)
    atom_positions_zyx, atom_ids, atom_b_factors = remove_hydrogens(
        atom_positions_zyx, atom_ids, atom_b_factors
    )

    # Scale the B-factors (now doing it after filtered unlike before)
    # NOTE: the 0.25 is strange but keeping like cisTEM for now
    print("Calculating scattering parameters")  # TODO: Move to logging
    atom_b_factors = 0.25 * (atom_b_factors * b_scaling + added_B)

    #################################################
    ### Atomwise scattering potential calculation ###
    #################################################

    print("Simulating atomwise scattering potentials")
    scattering_results = simulate_atomwise_scattering_potentials(
        atom_positions_zyx=atom_positions_zyx,
        atom_ids=atom_ids,
        atom_b_factors=atom_b_factors,
        sim_pixel_spacing=sim_pixel_spacing,
        sim_volume_shape=sim_volume_shape,
        lead_term=lead_term,
        upsampling=upsampling,
    )

    neighborhood_potentials = scattering_results["neighborhood_potentials"]
    voxel_positions = scattering_results["voxel_positions"]
    upsampled_shape = scattering_results["upsampled_shape"]
    upsampled_pixel_size = scattering_results["upsampled_pixel_size"]
    upsampling = scattering_results["upsampling"]

    # Calculate the upsampled volume
    upsampled_volume = torch.zeros(upsampled_shape, dtype=torch.float32, device=device)
    upsampled_volume = place_voxel_neighborhoods_in_volume(
        neighborhood_potentials=neighborhood_potentials,
        voxel_positions=voxel_positions,
        final_volume=upsampled_volume,
    )

    ############################################
    ### Additional post-simulation filtering ###
    ############################################

    # Convert to Fourier space for filtering
    upsampled_volume = torch.fft.fftshift(upsampled_volume, dim=(-3, -2, -1))
    upsampled_volume_FFT = torch.fft.rfftn(upsampled_volume, dim=(-3, -2, -1))

    final_volume = apply_simulation_filters(
        upsampled_volume_rfft=upsampled_volume_FFT,
        upsampled_shape=upsampled_shape,
        upsampled_pixel_size=upsampled_pixel_size,
        upsampling=upsampling,
        final_shape=sim_volume_shape,
        dose_weighting=dose_weighting,
        num_frames=num_frames,
        fluence_per_frame=fluence_per_frame,
        dose_B=dose_B,
        modify_signal=modify_signal,
        apply_dqe=apply_dqe,
        mtf_filename=mtf_filename,
    )

    print("Writing mrc file")
    tensor_to_mrc(
        output_filename=output_filename,
        final_volume=final_volume,
        sim_pixel_spacing=sim_pixel_spacing,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Finished simulation")
    print(f"Total simulation time: {minutes} minutes {seconds} seconds")
