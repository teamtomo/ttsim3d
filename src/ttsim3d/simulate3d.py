"""The main simulation function."""

import multiprocessing as mp
import time
from typing import Optional

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
    get_scattering_parameters,
    get_scattering_potential_of_voxel,
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


def process_atom_batch(
    batch_args: tuple,
) -> torch.Tensor:
    """
    Process a batch of atoms to calculate the scattering potential in parallel.

    Args:
        batch_args: Tuple containing the arguments for the batch.

    Returns
    -------
        torch.Tensor: The local volume grid for the batch.
    """
    try:
        # Unpack the tuple correctly
        (
            atom_indices_batch,
            atom_dds_batch,
            bPlusB_batch,
            atoms_id_filtered_batch,
            voxel_offsets_flat,
            upsampled_shape,
            upsampled_pixel_size,
            lead_term,
            scattering_params_a,
        ) = batch_args

        # Move tensors to CPU and ensure they're contiguous
        atom_indices_batch = atom_indices_batch.cpu().contiguous()
        atom_dds_batch = atom_dds_batch.cpu().contiguous()
        voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()

        # Initialize local volume grid for this batch
        local_volume = torch.zeros(upsampled_shape, device="cpu")

        # Add debug print to verify data
        print(f"Processing batch of size {len(atom_indices_batch)}")

        # offset_test = upsampled_pixel_size/2
        # Process each atom in the batch
        for i in range(len(atom_indices_batch)):
            atom_pos = atom_indices_batch[i]
            atom_dds = atom_dds_batch[i]
            atom_id = atoms_id_filtered_batch[i]

            # Calculate voxel positions relative to atom center
            voxel_positions = (
                atom_pos.view(1, 3) + voxel_offsets_flat
            )  # indX/Y/Z equivalent

            # print(voxel_positions.shape)
            # Check bounds for each dimension separately
            valid_z = (voxel_positions[:, 0] >= 0) & (
                voxel_positions[:, 0] < upsampled_shape[0]
            )
            valid_y = (voxel_positions[:, 1] >= 0) & (
                voxel_positions[:, 1] < upsampled_shape[1]
            )
            valid_x = (voxel_positions[:, 2] >= 0) & (
                voxel_positions[:, 2] < upsampled_shape[2]
            )
            valid_mask = valid_z & valid_y & valid_x

            if valid_mask.any():
                # Calculate coordinates relative to atom center
                relative_coords = (
                    voxel_positions[valid_mask] - atom_pos - atom_dds - PIXEL_OFFSET
                ) * upsampled_pixel_size
                coords1 = relative_coords
                coords2 = relative_coords + upsampled_pixel_size

                # Calculate potentials for valid positions
                potentials = get_scattering_potential_of_voxel(
                    zyx_coords1=coords1,
                    zyx_coords2=coords2,
                    bPlusB=bPlusB_batch[i],
                    atom_id=atom_id,
                    lead_term=lead_term,
                    scattering_params_a=scattering_params_a,  # Pass the parameters
                )

                # Get valid voxel positions
                valid_positions = voxel_positions[valid_mask].long()

                # Update local volume
                local_volume[
                    valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]
                ] += potentials
    except Exception as e:
        print(f"Error in process_atom_batch: {e!s}")
        raise e

    return local_volume


def process_atoms_parallel(
    atom_indices: torch.Tensor,
    atom_dds: torch.Tensor,
    bPlusB: torch.Tensor,
    scattering_params_a: dict,
    atoms_id_filtered: list[str],
    voxel_offsets_flat: torch.Tensor,
    upsampled_shape: tuple[int, int, int],
    upsampled_pixel_size: float,
    lead_term: float,
    n_cores: int = 1,
) -> torch.Tensor:
    """
    Scattering potential of atoms in parallel using cpu multiprocessing.

    Args:
        atom_indices: The indices of the atoms.
        atom_dds: The offset from the edge of the voxel.
        bPlusB: The sum of the B factors from scattering and pdb file.
        scattering_params_a: The 'a' scattering parameters.
        atoms_id_filtered: The list of atom IDs (no H).
        voxel_offsets_flat: The flattened voxel offsets for the neighborhood.
        upsampled_shape: The shape of the upsampled volume.
        upsampled_pixel_size: The pixel size of the upsampled volume.
        lead_term: The lead term for the calculation.
        n_cores: The number of CPU cores to use.

    Returns
    -------
        torch.Tensor: The final volume grid.
    """
    # Ensure all inputs are on CPU and contiguous
    atom_indices = atom_indices.cpu().contiguous()
    atom_dds = atom_dds.cpu().contiguous()
    voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()

    # Convert pandas Series to list if necessary
    if hasattr(atoms_id_filtered, "tolist"):
        atoms_id_filtered = atoms_id_filtered.tolist()

    num_atoms = len(atom_indices)
    batch_size = max(1, num_atoms // (n_cores))  # Divide work into smaller batches

    print(f"Processing {num_atoms} atoms in batches of {batch_size}")

    # Prepare batches
    batches = []
    for start_idx in range(0, num_atoms, batch_size):
        end_idx = min(start_idx + batch_size, num_atoms)
        batch_args = (
            atom_indices[start_idx:end_idx],
            atom_dds[start_idx:end_idx],
            bPlusB[start_idx:end_idx],
            atoms_id_filtered[start_idx:end_idx],
            voxel_offsets_flat,
            upsampled_shape,
            upsampled_pixel_size,
            lead_term,
            scattering_params_a,
        )
        batches.append(batch_args)

    # Process batches in parallel
    with mp.Pool(n_cores) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_atom_batch, batches)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * batch_size} atoms of {num_atoms}")

    # Combine results
    final_volume = torch.zeros(upsampled_shape, device="cpu")
    for result in results:
        final_volume += result

    return final_volume


def process_device_atoms(
    args: tuple,
) -> torch.Tensor:
    """
    Process atoms for a single device (gpu or cpu) in parallel.

    Args:
        args: Tuple containing the arguments for the device.

    Returns
    -------
        torch.Tensor: The final volume grid for the device.
    """
    (
        device_atom_indices,
        device_atom_dds,
        device_bPlusB,
        device_atoms_id,
        scattering_params_a,
        voxel_offsets_flat,
        upsampled_shape,
        upsampled_pixel_size,
        lead_term,
        device,
        n_cpu_cores,
    ) = args

    print(f"\nProcessing atoms on {device}")

    if device.type == "cuda":
        print("Not done this yet!")
    else:
        volume_grid = process_atoms_parallel(
            atom_indices=device_atom_indices,
            atom_dds=device_atom_dds,
            bPlusB=device_bPlusB,
            scattering_params_a=scattering_params_a,
            atoms_id_filtered=device_atoms_id,
            voxel_offsets_flat=voxel_offsets_flat,
            upsampled_shape=upsampled_shape,
            upsampled_pixel_size=upsampled_pixel_size,
            lead_term=lead_term,
            n_cores=n_cpu_cores,
        )

    return volume_grid


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

    # It is called by ttsim3d.py with all the inputs

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

    # Now divide into chunks for parallel processing

    # atoms_per_device = len(atoms_id_filtered) // num_devices
    device_outputs = []
    # device_args = []
    if devices[0].type == "cpu":
        # If CPU only, use the original parallel processing directly
        volume_grid = process_atoms_parallel(
            atom_indices=atom_indices,
            atom_dds=atom_dds,
            bPlusB=total_b_param,
            scattering_params_a=scattering_params_a,
            atoms_id_filtered=atoms_id_filtered,
            voxel_offsets_flat=voxel_offsets_flat,
            upsampled_shape=upsampled_shape,
            upsampled_pixel_size=upsampled_pixel_size,
            lead_term=lead_term,
            n_cores=n_cpu_cores,
        )
        device_outputs = [volume_grid]
    else:
        print("Not done this yet!")

    # Combine results from all devices
    main_device = devices[0]
    final_volume = torch.zeros(upsampled_shape, device=main_device)
    for volume in device_outputs:
        final_volume += volume.to(main_device)

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
    # apply dqe
    # I should really apply the mtf after Fourier cropping
    # cisTEM does it before
    """
    if apply_dqe:
        mtf_frequencies, mtf_amplitudes = read_mtf(
            file_path=mtf_filename
        )
        mtf = make_mtf_grid(
            image_shape=final_volume.shape,
            mtf_frequencies=mtf_frequencies, #1D tensor
            mtf_amplitudes=mtf_amplitudes, #1D tensor
            rfft=True,
            fftshift=False,
        )
        final_volume_FFT *= mtf
    """
    # fourier crop back to desired output size

    if upsampling > 1:
        final_volume_FFT = fourier_rescale_3d_force_size(
            volume_fft=final_volume_FFT,
            volume_shape=final_volume.shape,
            target_size=sim_volume_shape[0],
            rfft=True,
            fftshift=False,
        )

    if apply_dqe:
        """
        mtf = get_dqe_parameterized(
            image_shape=final_volume.shape,
            pixel_size=upsampled_pixel_size,
            rfft=True,
            fftshift=False,
        )
        """
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
