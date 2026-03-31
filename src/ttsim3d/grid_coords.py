"""Deals with grid coordinates."""

import numbers
from typing import Union

import einops
import numpy as np
import torch
from torch_fourier_rescale import fourier_rescale_rfft_3d


def get_upsampling(
    wanted_pixel_size: float, wanted_output_size: int, max_size: int = 1536
) -> int:
    """Calculate the upsampling factor for the simulation volume.

    Parameters
    ----------
    wanted_pixel_size : float
        The pixel size in Angstroms.
    wanted_output_size : float
        The output size of the cubic volume.
    max_size : int
        Optional maximum size of the volume. Default is 1536.

    Returns
    -------
    int
        The upsampling factor.
    """
    if wanted_pixel_size > 1.5 and wanted_output_size * 4 < max_size:
        # print("Oversampling your 3d by a factor of 4 for calculation.")
        return 4

    if 0.75 < wanted_pixel_size <= 1.5 and wanted_output_size * 2 < max_size:
        # print("Oversampling your 3d by a factor of 2 for calculation.")
        return 2

    return 1


def get_atom_voxel_indices(
    atom_zyx: torch.Tensor,
    upsampled_pixel_size: float,
    upsampled_shape: tuple[int, int, int],
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the voxel indices of the atoms.

    Parameters
    ----------
    atom_zyx : torch.Tensor
        The atom coordinates in Angstroms.
    upsampled_pixel_size : float
        The pixel size in Angstroms.
    upsampled_shape : tuple[int, int, int]
        The shape of the upsampled volume.
    offset : float
        Optional voxel edge offset in units of voxels. Default is 0.5.

    Returns
    -------
    tuple[torch.Tensor,torch.Tensor]
        The voxel indices and the offset from the edge of the voxel.
    """
    # Move to device
    device = atom_zyx.device
    shape_tensor = torch.tensor(upsampled_shape, device=device)
    offset_tensor = torch.tensor(offset, device=device)
    pixel_size_tensor = torch.tensor(upsampled_pixel_size, device=device)

    origin_idx = (
        shape_tensor[0] / 2,
        shape_tensor[1] / 2,
        shape_tensor[2] / 2,
    )
    origin_idx_tensor = torch.tensor(origin_idx, device=device)
    this_coords = (
        (atom_zyx / pixel_size_tensor) + origin_idx_tensor.unsqueeze(0) + offset_tensor
    )
    atom_indices = torch.floor(this_coords)  # these are the voxel indices
    atom_dds = (
        this_coords - atom_indices - offset
    )  # this is offset from the edge of the voxel

    return atom_indices, atom_dds


def get_size_neighborhood_cistem(
    mean_b_factor: float, upsampled_pixel_size: float
) -> int:
    """Calculate the size of the neighborhood of voxels (mirrors cisTEM).

    Parameters
    ----------
    mean_b_factor : float
        The mean B factor over all the atoms.
    upsampled_pixel_size : float
        The pixel size in Angstroms.

    Returns
    -------
    int
        The size (number of voxels in one direction) of the neighborhood.
    """
    tmp = 0.4 * (0.6 * mean_b_factor) ** 0.5 + 0.2
    tmp = torch.round(tmp / upsampled_pixel_size)

    return int(tmp + 1)


def get_voxel_neighborhood_offsets(
    mean_b_factor: float, upsampled_pixel_size: float
) -> torch.Tensor:
    """Offset arrays for the voxel neighborhood.

    Calculate the offsets of the voxel neighborhood. Returned as a flat tensor
    with shape (n^3, 3) where n is the size of the neighborhood in one dimension.

    Parameters
    ----------
    mean_b_factor : float
        The mean B factor of the atoms.
    upsampled_pixel_size : float
        The pixel size in Angstroms.

    Returns
    -------
    torch.Tensor
        The offsets of the voxel neighborhood.

    """
    device = mean_b_factor.device if isinstance(mean_b_factor, torch.Tensor) else "cpu"
    # Get the size of the voxel neighbourhood to calculate the potential of each atom
    size_neighborhood = get_size_neighborhood_cistem(
        mean_b_factor, upsampled_pixel_size
    )
    neighborhood_range = torch.arange(
        -size_neighborhood, size_neighborhood + 1, device=device
    )
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
def fourier_rescale_3d(
    volume_fft: torch.Tensor,
    volume_shape: tuple[int, int, int],
    upsampled_pixel_size: Union[float, tuple[float, float, float]],
    target_pixel_size: Union[float, tuple[float, float, float]],
) -> torch.Tensor:
    """
    Crop a 3D Fourier-transformed volume to a specific target size.

    Parameters
    ----------
    volume_fft : torch.Tensor
        The Fourier-transformed volume.
    volume_shape : tuple[int, int, int]
        The original shape of the volume.
    upsampled_pixel_size : float
        The pixel size in Angstroms.
    target_pixel_size : float
        The pixel size in Angstroms.

    Returns
    -------
    cropped_fft_shifted_back : torch.Tensor
        The cropped fft
    """
    if isinstance(upsampled_pixel_size, int | float | numbers.Real):
        upsampled_pixel_size = (
            upsampled_pixel_size,
            upsampled_pixel_size,
            upsampled_pixel_size,
        )
    if isinstance(target_pixel_size, int | float | numbers.Real):
        target_pixel_size = (target_pixel_size, target_pixel_size, target_pixel_size)
    # fft shift to center the dft
    dft = torch.fft.fftshift(volume_fft, dim=(-3, -2))
    # Fourier crop
    dft, new_nyquist, new_shape = fourier_rescale_rfft_3d(
        dft=dft,
        image_shape=volume_shape,
        source_spacing=upsampled_pixel_size,
        target_spacing=target_pixel_size,
    )

    # ifft shift back
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))

    # Calculate new spacing after rescaling
    source_spacing = np.array(upsampled_pixel_size, dtype=np.float32)
    new_nyquist = np.array(new_nyquist, dtype=np.float32)
    new_spacing = 1 / (2 * new_nyquist * (1 / source_spacing))

    return dft, new_spacing, new_shape
