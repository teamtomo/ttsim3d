"""The main simulation function."""

import time
from typing import Optional

import einops
import mrcfile
import numpy as np
import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid, read_mtf

from ttsim3d.device_handler import get_cpu_cores, select_gpus
from ttsim3d.grid_coords import get_size_neighborhood_cistem, get_upsampling
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


# This will definitely be moved to a different program
def fourier_rescale_3d_force_size(
    volume_fft: torch.Tensor,
    volume_shape: tuple[int, int, int],
    target_size: int,
    rfft: bool = True,
    fftshift: bool = False,
) -> torch.Tensor:
    """
    Crop a 3D Fourier-transformed volume to a specific target size.

    Parameters
    ----------
    volume_fft: torch.Tensor
        The Fourier-transformed volume.
    volume_shape: tuple[int, int, int]
        The original shape of the volume.
    target_size: int
        The target size of the cropped volume.
    rfft: bool
        Whether the input is a real-to-complex Fourier Transform.
    fftshift: bool
        Whether the zero frequency is shifted to the center.

    Returns
    -------
    - cropped_fft_shifted_back (torch.Tensor): The cropped fft
    """
    # Ensure the target size is even
    assert target_size > 0, "Target size must be positive."

    # Get the original size of the volume
    assert (
        volume_shape[0] == volume_shape[1] == volume_shape[2]
    ), "Volume must be cubic."

    # Step 1: Perform real-to-complex Fourier Transform (rfftn)
    # and shift the zero frequency to the center
    if not fftshift:
        volume_fft = torch.fft.fftshift(
            volume_fft, dim=(-3, -2, -1)
        )  # Shift along first two dimensions only

    # Calculate the dimensions of the rfftn output
    rfft_size_z, rfft_size_y, rfft_size_x = volume_fft.shape

    # Calculate cropping indices for each dimension
    center_z = rfft_size_z // 2
    center_y = rfft_size_y // 2
    center_x = rfft_size_x // 2

    # Define the cropping ranges
    crop_start_z = int(center_z - target_size // 2)
    crop_end_z = int(crop_start_z + target_size)
    crop_start_y = int(center_y - target_size // 2)
    crop_end_y = int(crop_start_y + target_size)
    crop_start_x = int(center_x - target_size // 2)
    crop_end_x = int(
        target_size // 2 + 1
    )  # Crop from the high-frequency end only along the last dimension

    # Step 2: Crop the Fourier-transformed volume
    cropped_fft = torch.zeros_like(volume_fft)
    if rfft:
        cropped_fft = volume_fft[
            crop_start_z:crop_end_z, crop_start_y:crop_end_y, -crop_end_x:
        ]
    else:
        crop_end_x = int(crop_start_x + target_size)
        cropped_fft = volume_fft[
            crop_start_z:crop_end_z, crop_start_y:crop_end_y, crop_start_x:crop_end_x
        ]

    # Step 3: Inverse shift and apply the inverse rFFT to return to real space
    cropped_fft_shifted_back = torch.fft.ifftshift(cropped_fft, dim=(-3, -2))

    return cropped_fft_shifted_back


def process_atoms_single_thread(
    atom_indices: torch.Tensor,
    atom_dds: torch.Tensor,
    bPlusB: torch.Tensor,
    scattering_params_a: torch.Tensor,
    voxel_offsets_flat: torch.Tensor,
    upsampled_shape: tuple[int, int, int],
    upsampled_pixel_size: float,
    lead_term: float,
    n_cores: int = 1,
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
    n_cpu_cores: int = 1,  # -1 id get automatically
    gpu_ids: Optional[list[int]] = None,  # [-999] cpu, [-1] auto, [0, 1] etc=gpuid
    num_gpus: int = 1,
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
        n_cpu_cores: The number of CPU cores.
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
    if gpu_ids == [-999]:  # Special case for CPU-only
        devices = [torch.device("cpu")]
    else:
        devices = select_gpus(gpu_ids, num_gpus)
    if devices[0].type == "cpu":
        if n_cpu_cores == -1:
            n_cpu_cores = get_cpu_cores()
    print(f"Using devices: {[str(device) for device in devices]}")

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
    # Get the centre if the upsampled volume
    origin_idx = (
        upsampled_shape[0] / 2,
        upsampled_shape[1] / 2,
        upsampled_shape[2] / 2,
    )
    # Get the size of the voxel neighbourhood to calculate the potential of each atom
    size_neighborhood = get_size_neighborhood_cistem(
        mean_b_factor, upsampled_pixel_size
    )
    neighborhood_range = torch.arange(-size_neighborhood, size_neighborhood + 1)
    # Create coordinate grids for the neighborhood
    sz, sy, sx = torch.meshgrid(
        neighborhood_range, neighborhood_range, neighborhood_range, indexing="ij"
    )
    voxel_offsets = torch.stack([sz, sy, sx])  # (3, n, n, n)
    # Flatten while preserving the relative positions
    voxel_offsets_flat = voxel_offsets.reshape(3, -1).T  # (n^3, 3)
    # Calculate the pixel coordinates of each atom
    this_coords = (
        (atoms_zyx_filtered / upsampled_pixel_size)
        + torch.tensor(origin_idx).unsqueeze(0)
        + PIXEL_OFFSET
    )
    atom_indices = torch.floor(this_coords)  # these are the voxel indices
    atom_dds = (
        this_coords - atom_indices - PIXEL_OFFSET
    )  # this is offset from the edge of the voxel

    final_volume = process_atoms_single_thread(
        atom_indices=atom_indices,
        atom_dds=atom_dds,
        bPlusB=total_b_param,
        scattering_params_a=total_a_param,
        voxel_offsets_flat=voxel_offsets_flat,
        upsampled_shape=upsampled_shape,
        upsampled_pixel_size=upsampled_pixel_size,
        lead_term=lead_term,
        n_cores=n_cpu_cores,
        device=torch.device("cpu"),
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
