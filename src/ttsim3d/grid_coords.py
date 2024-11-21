"""Deals with grid coordinates."""

import einops
import torch


def get_upsampling(
    wanted_pixel_size: float, wanted_output_size: int, max_size: int = 1536
) -> int:
    """
    Calculate the upsampling factor for the simulation volume.

    Args:
        wanted_pixel_size: The pixel size in Angstroms.
        wanted_output_size: The output size of the 3D volume.
        max_size: The maximum size of the 3D volume.

    Returns
    -------
        int: The upsampling factor.
    """
    if wanted_pixel_size > 1.5 and wanted_output_size * 4 < max_size:
        print("Oversampling your 3d by a factor of 4 for calculation.")
        return 4

    if 0.75 < wanted_pixel_size <= 1.5 and wanted_output_size * 2 < max_size:
        print("Oversampling your 3d by a factor of 2 for calculation.")
        return 2

    return 1


def get_atom_voxel_indices(
    atom_zyx: torch.Tensor,
    upsampled_pixel_size: float,
    upsampled_shape: tuple[int, int, int],
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the voxel indices of the atoms.

    Args:
        atom_zyx: The atom coordinates in Angstroms.
        upsampled_pixel_size: The pixel size in Angstroms.
        upsampled_shape: The shape of the upsampled volume.

    Returns
    -------
        tuple[torch.Tensor,torch.Tensor]: The voxel indices + offsets of the atoms.
    """
    origin_idx = (
        upsampled_shape[0] / 2,
        upsampled_shape[1] / 2,
        upsampled_shape[2] / 2,
    )
    this_coords = (
        (atom_zyx / upsampled_pixel_size)
        + torch.tensor(origin_idx).unsqueeze(0)
        + offset
    )
    atom_indices = torch.floor(this_coords)  # these are the voxel indices
    atom_dds = (
        this_coords - atom_indices - offset
    )  # this is offset from the edge of the voxel

    return atom_indices, atom_dds


def get_size_neighborhood_cistem(
    mean_b_factor: float, upsampled_pixel_size: float
) -> int:
    """
    Calculate the size of the neighborhood of voxels.

    Args:
        mean_b_factor: The mean B factor of the atoms.
        upsampled_pixel_size: The pixel size in Angstroms.

    Returns
    -------
        int: The size of the neighborhood.
    """
    return int(
        1
        + torch.round((0.4 * (0.6 * mean_b_factor) ** 0.5 + 0.2) / upsampled_pixel_size)
    )


def get_voxel_neighborhood_offsets(
    mean_b_factor: float, upsampled_pixel_size: float
) -> torch.Tensor:
    """Offset arrays for the voxel neighborhood.

    Calculate the offsets of the voxel neighborhood. Returned as a flat tensor
    with shape (n^3, 3) where n is the size of the neighborhood in one dimension.

    Args:
        mean_b_factor: The mean B factor of the atoms.
        upsampled_pixel_size: The pixel size in Angstroms.

    Returns
    -------
        torch.Tensor: The offsets of the voxel neighborhood.

    """
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
    voxel_offsets_flat = einops.rearrange(
        voxel_offsets, "c x y z -> (x y z) c"
    )  # (n^3, 3)

    return voxel_offsets_flat


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
