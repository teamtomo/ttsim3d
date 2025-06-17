"""Handles cpu/gpu device selection."""

from typing import Union

import psutil
import torch


def calculate_batches(
    setup_results: dict,
    upsampled_volume: torch.Tensor,
    memory_buffer: float = 0.8,
    fudge_factor: int = 4,
) -> tuple[int, int]:
    """Calculate the number of batches based on available memory.

    Parameters
    ----------
    setup_results : dict
        Contains the setup results including atom indices and voxel offsets.
    upsampled_volume : torch.Tensor
        The upsampled volume tensor.
    memory_buffer : float, optional
        The fraction of available memory to use, by default 0.8.
    fudge_factor : int, optional
        A factor to account for additional memory usage, by default 4.

    Returns
    -------
    int
        The number of batches.
    """
    num_el = (
        setup_results["atom_indices"].shape[0]
        * setup_results["voxel_offsets_flat"].shape[0]
        * setup_results["voxel_offsets_flat"].shape[1]
    )
    memory_size_needed = 4 * (
        num_el * setup_results["voxel_offsets_flat"].element_size()
    ) + 2 * (num_el * upsampled_volume.element_size())
    memory_size_needed += upsampled_volume.numel() * upsampled_volume.element_size()
    memory_size_needed *= fudge_factor

    # BUG: This looks at the system memory, not the GPU memory. This function
    # somehow needs access to which GPU device is being used calculate from there.
    available_memory = psutil.virtual_memory().available

    atoms_per_batch = max(
        1,
        int(
            (available_memory * memory_buffer)
            / memory_size_needed
            * setup_results["atom_indices"].shape[0]
        ),
    )
    num_batches = max(
        1,
        (setup_results["atom_indices"].shape[0] + atoms_per_batch - 1)
        // atoms_per_batch,
    )

    return num_batches, atoms_per_batch


def get_device(
    device: Union[str, int, list[str], list[int]] = "all",
) -> list[torch.device]:
    """Get the appropriate torch device(s) based on availability and user preference.

    NOTE: This function does not itself raise an error when no CUDA devices are found,
    rather it relies on PyTorch for error raising. Additionally, it does not
    fall back to CPU if no CUDA devices are available.

    Parameters
    ----------
    device : Union[str, int, list[str], list[int]]
        Device selection preference:
        - 'cpu': Use CPU
        - 'all': Use all available CUDA devices
        - 'cuda:N' or int N: Use specific CUDA device N
        - ['cuda:0', 'cuda:2'] or [0, 2]: Use specific CUDA devices in the list.

    Returns
    -------
    list[torch.device]
        A list of torch devices based on the selection. For single device requests,
        a single-element list is returned.
    """
    # Case for CPU execution
    if device == "cpu":
        return [torch.device("cpu")]

    devices: list[torch.device] = []
    if device == "all":
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    elif isinstance(device, list):
        devices = [
            torch.device(f"cuda:{i}") if isinstance(i, int) else torch.device(i)
            for i in device
        ]
    elif isinstance(device, int):
        devices = [torch.device(f"cuda:{device}")]
    elif isinstance(device, str):
        devices = [torch.device(device)]
    else:
        raise ValueError(
            f"Invalid device selection: {device}. Use 'cpu', 'all', an int, list of "
            "ints, or list of device strings."
        )

    assert len(devices) > 0, "No valid devices found."

    return devices
