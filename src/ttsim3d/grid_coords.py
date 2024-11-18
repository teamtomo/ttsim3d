"""Deals with grid coordinates."""

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
