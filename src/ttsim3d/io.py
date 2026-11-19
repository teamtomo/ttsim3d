"""File output for simulated volumes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mrcfile  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    import os

    import torch


def write_mrc(
    path: str | os.PathLike[str],
    volume: torch.Tensor,
    pixel_spacing: float,
    *,
    overwrite: bool = True,
) -> None:
    """Write a 3D volume to an MRC file as float32.

    Parameters
    ----------
    path : str | os.PathLike
        Output file path.
    volume : torch.Tensor
        Volume with shape ``(d, h, w)`` (ZYX).
    pixel_spacing : float
        Isotropic voxel size in Angstroms, written to the header.
    overwrite : bool
        If True (default), overwrite an existing file.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {tuple(volume.shape)}")

    data = volume.detach().cpu().numpy().astype(np.float32, copy=False)
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(data)  # also updates the header dimensions and statistics
        mrc.voxel_size = pixel_spacing
