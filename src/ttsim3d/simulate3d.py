"""The main simulation function."""

import time
from typing import Optional

import einops
import mrcfile
import numpy as np
import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid, read_mtf

from ttsim3d.device_handler import select_gpu
from ttsim3d.grid_coords import (
    fourier_rescale_3d_force_size,
    get_atom_voxel_indices,
    get_upsampling,
    get_voxel_neighborhood_offsets,
)
from ttsim3d.pdb_handler import load_model, remove_hydrogens
from ttsim3d.scattering_potential import (
    calculate_relativistic_electron_wavelength,
    get_a_param,
    get_scattering_parameters,
    get_scattering_potential_of_voxel_batch,
    get_total_b_param,
)

BOND_SCALING_FACTOR = 1.043
PIXEL_OFFSET = 0.5


def process_atoms_single_thread(
    atom_indices: torch.Tensor,
    atom_dds: torch.Tensor,
    bPlusB: torch.Tensor,
    scattering_params_a: torch.Tensor,
    voxel_offsets_flat: torch.Tensor,
    upsampled_shape: tuple[int, int, int],
    upsampled_pixel_size: float,
    lead_term: float,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Process atoms in a single thread.

    Args:
        atom_indices: The indices of the atoms.
        atom_dds: The dds of the atoms.
        bPlusB: The B factor.
        scattering_params_a: The scattering parameters.
        voxel_offsets_flat: The flat voxel offsets.
        upsampled_shape: The upsampled shape.
        upsampled_pixel_size: The upsampled pixel size.
        lead_term: The lead term.
        n_cores: The number of cores.
        device: The device.

    Returns
    -------
        torch.Tensor: The final volume.
    """
    # Calculate voxel positions relative to atom center
    atom_pos = einops.rearrange(atom_indices, "a d -> a 1 d")
    voxel_offsets_flat = einops.rearrange(voxel_offsets_flat, "v d -> 1 v d")
    voxel_positions = atom_pos + voxel_offsets_flat  # (n_atoms, n^3, 3)

    # get relative coords
    atom_dds = einops.rearrange(atom_dds, "a d -> a 1 d")
    relative_coords = (
        voxel_positions - atom_pos - atom_dds - PIXEL_OFFSET
    ) * upsampled_pixel_size
    coords1 = relative_coords
    coords2 = relative_coords + upsampled_pixel_size

    potentials = get_scattering_potential_of_voxel_batch(
        zyx_coords1=coords1,
        zyx_coords2=coords2,
        bPlusB=bPlusB,
        lead_term=lead_term,
        scattering_params_a=scattering_params_a,  # Pass the parameters
    )  # shape(a, v)

    # add to volume
    index_positions = voxel_positions.long()
    final_volume = torch.zeros(upsampled_shape, device=device)
    final_volume.index_put_(
        (index_positions[:, :, 0], index_positions[:, :, 1], index_positions[:, :, 2]),
        potentials,
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
        n_cores: The number of CPU cores.
        gpu_ids: The list of GPU IDs.
        num_gpus: The number of GPUs.
        modify_signal: The signal modification factor.

    Returns
    -------
        None
    """
    # This is the main program
    start_time = time.time()
    # Get the wavelength from the beam energy
    wavelength_A = (
        calculate_relativistic_electron_wavelength(beam_energy_kev * 1000) * 1e10
    )
    # Get lead term, call it something better and move it out elsewhere
    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)
    # get the scattering parameters
    scattering_params_a, scattering_params_b = get_scattering_parameters()

    # Select devices
    if gpu_id == -999:  # Special case for CPU-only
        device = torch.device("cpu")
    else:
        device = select_gpu(gpu_id)
    print(f"Using device: {device!s}")

    # Then load pdb (a separate file) and get non-H atom
    # list with zyx coords and isotropic b factors
    atoms_zyx, atoms_id, atoms_b_factor = load_model(pdb_filename)
    atoms_zyx_filtered, atoms_id_filtered, atoms_b_factor_filtered = remove_hydrogens(
        atoms_zyx, atoms_id, atoms_b_factor
    )
    # Scale the B-factors (now doing it after filtered unlike before)
    # the 0.25 is strange but keeping like cisTEM for now
    atoms_b_factor_scaled = 0.25 * (atoms_b_factor_filtered * b_scaling + added_B)
    mean_b_factor = torch.mean(atoms_b_factor_scaled)
    # Get the B parameter for each atom plus scattering parameter B
    total_b_param = get_total_b_param(
        scattering_params_b, atoms_id_filtered, atoms_b_factor_scaled
    )
    # get the scattering parameters 'a' for each atom in a tensor
    total_a_param = get_a_param(scattering_params_a, atoms_id_filtered)

    # Set up the simulation volume - push this out into a separate file grid_coords.py
    # Start with upsampling to improve accuracy
    upsampling = (
        get_upsampling(sim_pixel_spacing, sim_volume_shape[0], max_size=1536)
        if upsampling == -1
        else upsampling
    )
    upsampled_pixel_size = sim_pixel_spacing / upsampling
    upsampled_shape = tuple(np.array(sim_volume_shape) * upsampling)

    # Calculate the voxel coordinates of each atom
    atom_indices, atom_dds = get_atom_voxel_indices(
        atom_zyx=atoms_zyx_filtered,
        upsampled_pixel_size=upsampled_pixel_size,
        upsampled_shape=upsampled_shape,
        offset=PIXEL_OFFSET,
    )

    # Get the voxel offsets for the neighborhood around the atom voxel
    voxel_offsets_flat = get_voxel_neighborhood_offsets(
        mean_b_factor=mean_b_factor,
        upsampled_pixel_size=upsampled_pixel_size,
    )

    # Calcaulte the scattering potential
    final_volume = torch.zeros(upsampled_shape, dtype=torch.float32, device=device)
    final_volume = process_atoms_single_thread(
        atom_indices=atom_indices,
        atom_dds=atom_dds,
        bPlusB=total_b_param,
        scattering_params_a=total_a_param,
        voxel_offsets_flat=voxel_offsets_flat,
        upsampled_shape=upsampled_shape,
        upsampled_pixel_size=upsampled_pixel_size,
        lead_term=lead_term,
        device=device,
    )

    # Convert to Fourier space for filtering
    final_volume = torch.fft.fftshift(final_volume, dim=(-3, -2, -1))
    final_volume_FFT = torch.fft.rfftn(final_volume, dim=(-3, -2, -1))
    # Dose weight
    if dose_weighting:
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

    # Write now for testing
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(cropped_volume.cpu().numpy())
        mrc.voxel_size = (sim_pixel_spacing, sim_pixel_spacing, sim_pixel_spacing)
        # Populate more of the metadata...

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total simulation time: {minutes} minutes {seconds} seconds")
