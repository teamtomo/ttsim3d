"""Handles cpu/gpu device selection."""

import warnings
from typing import Optional, Union

import psutil
import torch

MULTI_GPU_WARNING = (
    "Multiple GPU devices were selected, but multi-device execution is not currently "
    "supported by ttsim3d. Defaulting to the first device in the list..."
)


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
) -> Union[torch.device, list[torch.device]]:
    """Get the appropriate torch device(s) based on availability and user preference.

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
    Union[torch.device, list[torch.device]]
        The selected compute device(s). Returns a list for multi-GPU selection.
    """
    # Case for CPU execution
    if device == "cpu":
        return torch.device("cpu")

    # Case for using all available CUDA devices
    if device == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available.")

        tmp = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        # NOTE: Multi-gpu execution is not currently supported
        warnings.warn(MULTI_GPU_WARNING, stacklevel=2)
        return tmp[0]

    # Case for list of integers
    if isinstance(device, list) and all(isinstance(d, int) for d in device):
        tmp = [torch.device(f"cuda:{i}") for i in device]

        # NOTE: Multi-gpu execution is not currently supported
        warnings.warn(MULTI_GPU_WARNING, stacklevel=2)
        return tmp[0]

    # Case for list of strings
    if isinstance(device, list) and all(isinstance(d, str) for d in device):
        tmp = [torch.device(d) for d in device]

        # NOTE: Multi-gpu execution is not currently supported
        warnings.warn(MULTI_GPU_WARNING, stacklevel=2)
        return tmp[0]

    # Case for single integer
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")

    # Case for single string
    if isinstance(device, str):
        return torch.device(device)

    raise ValueError(
        f"Invalid device selection: {device}. Use 'cpu', 'all', an int, list of ints, "
        "or list of device strings"
    )


def select_gpu(
    gpu_id: Optional[int] = None,
) -> torch.device:
    """
    Select multiple GPU devices based on IDs or available memory.

    Args:
        gpu_id: Specific GPU ID to use.
                If None, selects GPU with most available memory.

    Returns
    -------
        torch.device: Selected GPU device or CPU if no GPU available
    """
    # Check if you can actually use a cuda gpu
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        return [torch.device("cpu")]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPU available, using CPU")
        return [torch.device("cpu")]

    # If specific GPUs requested, validate and return them
    if gpu_id is not None:
        valid_device = None
        if gpu_id >= n_gpus:
            print(f"Requested GPU {gpu_id} not available. Max GPU ID is {n_gpus-1}")
        else:
            valid_device = torch.device(f"cuda:{gpu_id}")

        if valid_device is None:
            print("No valid GPUs specified. Using CPU")
            return [torch.device("cpu")]
        return valid_device

    # Find GPU with most available memory
    gpu_memory_available = []
    print("\nAvailable GPUs:")
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        available = total_memory - allocated_memory

        print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"  Total memory: {total_memory/1024**3:.1f} GB")
        print(f"  Available memory: {available/1024**3:.1f} GB")

        gpu_memory_available.append((i, available))

    # Select the GPU with the most available memory
    selected_gpu_idx, _ = max(gpu_memory_available, key=lambda x: x[1])
    selected_gpu = torch.device(f"cuda:{selected_gpu_idx}")

    print("\nSelected GPU:", selected_gpu)
    return selected_gpu
