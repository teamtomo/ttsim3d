"""For handling MRC files."""

import mrcfile
import numpy as np
import torch


def tensor_to_mrc(
    output_filename: str,
    final_volume: torch.Tensor,
    sim_pixel_spacing: float,
) -> None:
    """Write the final volume to an MRC file.

    NOTE: This function is only for 3D volumetric data; 2D data is not supported.

    Parameters
    ----------
    output_filename : str
        Path to the output MRC file.
    final_volume : torch.Tensor
        Volume information to write to the MRC file.
    sim_pixel_spacing :
        The pixel spacing in the simulation.

    Returns
    -------
    None
    """
    write_volume = final_volume.cpu().numpy()
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(write_volume)
        mrc.voxel_size = (sim_pixel_spacing, sim_pixel_spacing, sim_pixel_spacing)
        # Populate more of the metadata...
        # Header setup
        mrc.header.mode = 2  # Mode 2 is float32
        mrc.header.nx, mrc.header.ny, mrc.header.nz = write_volume.shape[
            ::-1
        ]  # Dimensions (x, y, z)
        mrc.header.mx, mrc.header.my, mrc.header.mz = write_volume.shape[
            ::-1
        ]  # Sampling grid size (same as data dimensions)
        cell_dimensions = [dim * sim_pixel_spacing for dim in write_volume.shape[::-1]]
        mrc.header.cella.x = cell_dimensions[0]  # X dimension in angstroms
        mrc.header.cella.y = cell_dimensions[1]  # Y dimension in angstroms
        mrc.header.cella.z = cell_dimensions[2]  # Z dimension in angstroms

        mrc.header.mapc = 1  # Columns correspond to x
        mrc.header.mapr = 2  # Rows correspond to y
        mrc.header.maps = 3  # Sections correspond to z

        mrc.header.dmin = write_volume.min()  # Minimum density value
        mrc.header.dmax = write_volume.max()  # Maximum density value
        mrc.header.dmean = write_volume.mean()  # Mean density value

        # Additional metadata
        mrc.update_header_from_data()  # Automatically populates remaining fields
        mrc.header.rms = np.std(write_volume)  # RMS deviation of the density values
