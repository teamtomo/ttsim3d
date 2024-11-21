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

# Sim3DConstants = namedtuple(
#     "Sim3DConstants",
#     "wavelength_A lead_term scattering_params_a scattering_params_b",
# )


# def process_atoms_single_thread(
#     atom_indices: torch.Tensor,
#     atom_dds: torch.Tensor,
#     bPlusB: torch.Tensor,
#     scattering_params_a: torch.Tensor,
#     voxel_offsets_flat: torch.Tensor,
#     upsampled_shape: tuple[int, int, int],
#     upsampled_pixel_size: float,
#     lead_term: float,
#     device: torch.device = None,
# ) -> torch.Tensor:
#     """
#     Process atoms in a single thread.

#     Args:
#         atom_indices: The indices of the atoms.
#         atom_dds: The dds of the atoms.
#         bPlusB: The B factor.
#         scattering_params_a: The scattering parameters.
#         voxel_offsets_flat: The flat voxel offsets.
#         upsampled_shape: The upsampled shape.
#         upsampled_pixel_size: The upsampled pixel size.
#         lead_term: The lead term.
#         n_cores: The number of cores.
#         device: The device.

#     Returns
#     -------
#         torch.Tensor: The final volume.
#     """
#     # a -> atom index with shape (n_atoms)
#     # d -> dimension with shape (3)
#     # v -> flattened voxel indices for neighborhood with shape (h*w*d)

#     # Calculate voxel positions relative to atom center
#     atom_pos = einops.rearrange(atom_indices, "a d -> a 1 d")
#     voxel_offsets_flat = einops.rearrange(voxel_offsets_flat, "v d -> 1 v d")
#     voxel_positions = atom_pos + voxel_offsets_flat  # (n_atoms, n^3, 3)

#     # Calculate relative coordinates of each voxel in the neighborhood
#     atom_dds = einops.rearrange(atom_dds, "a d -> a 1 d")
#     relative_coords = (
#         voxel_positions - atom_pos - atom_dds - PIXEL_OFFSET
#     ) * upsampled_pixel_size
#     coords1 = relative_coords
#     coords2 = relative_coords + upsampled_pixel_size

#     neighborhood_potentials = get_scattering_potential_of_voxel_batch(
#         zyx_coords1=coords1,  # shape(a, v, d)
#         zyx_coords2=coords2,  # shape(a, v, d)
#         atom_ids=atom_ids,
#         atom_b_factors=atom_b_factors,
#         lead_term=lead_term,
#     )

#     # add to volume
#     index_positions = voxel_positions.long()
#     final_volume = torch.zeros(upsampled_shape, device=device)
#     final_volume.index_put_(
#         (
#             index_positions[:, :, 0],
#             index_positions[:, :, 1],
#             index_positions[:, :, 2],
#         ),
#         potentials,
#         accumulate=True,
#     )
#     return final_volume


# def _setup_sim3d_constants(
#     beam_energy_kev: float, sim_pixel_spacing: float
# ) -> namedtuple:
#     """Returns the necessary constants for the simulation."""
#     # Relativistic electron beam wavelength in Angstroms
#     beam_energy_ev = beam_energy_kev * 1000
#     wavelength_m = calculate_relativistic_electron_wavelength(beam_energy_ev)
#     wavelength_A = wavelength_m * 1e10

#     # Lead term for the scattering potential
#     lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)

#     scattering_params_a, scattering_params_b = get_scattering_parameters()

#     return Sim3DConstants(
#         wavelength_A=wavelength_A,
#         lead_term=lead_term,
#         scattering_params_a=scattering_params_a,
#         scattering_params_b=scattering_params_b,
#     )


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
        pdb_filename: The filename of the PDB file.
        output_filename: The filename of the output MRC file.
        sim_volume_shape: The shape of the simulation volume.
        sim_pixel_spacing: The pixel spacing of the simulation volume.
        num_frames: The number of frames for the simulation.
        fluence_per_frame: The fluence per frame.
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

    # const = _setup_sim3d_constants(
    #     beam_energy_kev=beam_energy_kev,
    #     sim_pixel_spacing=sim_pixel_spacing,
    # )
    # wavelength_A = const.wavelength_A
    # lead_term = const.wavelength_A
    # scattering_params_a = const.wavelength_A
    # scattering_params_b = const.wavelength_A

    beam_energy_ev = beam_energy_kev * 1000
    wavelength_m = calculate_relativistic_electron_wavelength(beam_energy_ev)
    wavelength_A = wavelength_m * 1e10
    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)

    #########################################
    ### PDB model to coords/ids/b-factors ###
    #########################################

    # Then load pdb (a separate file) and get non-H atom
    # list with zyx coords and isotropic b factors
    print("Loading PDB model")  # TODO: Move to logging
    atom_pos_zyx, atom_ids, atom_b_factors = load_model(pdb_filename)
    atom_pos_zyx, atom_ids, atom_b_factors = remove_hydrogens(
        atom_pos_zyx, atom_ids, atom_b_factors
    )

    # Scale the B-factors (now doing it after filtered unlike before)
    # NOTE: the 0.25 is strange but keeping like cisTEM for now
    print("Calculating scattering parameters")  # TODO: Move to logging
    atom_b_factors = 0.25 * (atom_b_factors * b_scaling + added_B)
    mean_b_factor = torch.mean(atom_b_factors)

    ######################################
    ### Calculations on upsampled grid ###
    ######################################

    upsampling, upsampled_pixel_size, upsampled_shape = _setup_sim3d_upsampling(
        sim_pixel_spacing=sim_pixel_spacing,
        sim_volume_shape=sim_volume_shape,
        upsampling=upsampling,
    )

    atom_indices, atom_dds, voxel_offsets_flat = _setup_upsampling_coords(
        atom_positions_zyx=atom_pos_zyx,
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

    final_volume = torch.zeros(upsampled_shape, dtype=torch.float32, device=device)
    final_volume = place_voxel_neighborhoods_in_volume(
        neighborhood_potentials=neighborhood_potentials,
        voxel_positions=voxel_positions,
        final_volume=final_volume,
    )

    # # Calculate the scattering potential
    # print("Calculating scattering potential for all atoms")
    # final_volume = torch.zeros(
    #     upsampled_shape, dtype=torch.float32, device=device
    # )
    # final_volume = process_atoms_single_thread(
    #     atom_indices=atom_indices,
    #     atom_dds=atom_dds,
    #     bPlusB=total_b_param,
    #     scattering_params_a=total_a_param,
    #     voxel_offsets_flat=voxel_offsets_flat,
    #     upsampled_shape=upsampled_shape,
    #     upsampled_pixel_size=upsampled_pixel_size,
    #     lead_term=lead_term,
    #     device=device,
    # )

    ############################################
    ### Additional post-simulation filtering ###
    ############################################

    # Convert to Fourier space for filtering

    final_volume = torch.fft.fftshift(final_volume, dim=(-3, -2, -1))
    final_volume_FFT = torch.fft.rfftn(final_volume, dim=(-3, -2, -1))
    # Dose weight
    if dose_weighting:
        print("Dose weighting")
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=final_volume.shape,
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

            final_volume_FFT *= modification
        elif modify_signal == 2:
            final_volume_FFT *= dose_filter**0.5
        else:
            final_volume_FFT *= dose_filter

    # fourier crop back to desired output size
    if upsampling > 1:
        print("Fourier cropping back to desired size")
        final_volume_FFT = fourier_rescale_3d_force_size(
            volume_fft=final_volume_FFT,
            volume_shape=final_volume.shape,
            target_size=sim_volume_shape[0],
            rfft=True,
            fftshift=False,
        )

    # If I apply dqe before Fourier cropping like cisTEM the output is the same.
    # I apply it after Fourier cropping
    if apply_dqe:
        print("Applying a DQE")
        mtf_frequencies, mtf_amplitudes = read_mtf(file_path=mtf_filename)
        mtf = make_mtf_grid(
            image_shape=sim_volume_shape,
            mtf_frequencies=mtf_frequencies,  # 1D tensor
            mtf_amplitudes=mtf_amplitudes,  # 1D tensor
            rfft=True,
            fftshift=False,
        )
        final_volume_FFT *= mtf

    # inverse FFT
    cropped_volume = torch.fft.irfftn(
        final_volume_FFT,
        s=(sim_volume_shape[0], sim_volume_shape[0], sim_volume_shape[0]),
        dim=(-3, -2, -1),
    )
    cropped_volume = torch.fft.ifftshift(cropped_volume, dim=(-3, -2, -1))

    print("Writing mrc file")
    tensor_to_mrc(
        output_filename=output_filename,
        final_volume=cropped_volume,
        sim_pixel_spacing=sim_pixel_spacing,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Finished simulation")
    print(f"Total simulation time: {minutes} minutes {seconds} seconds")
