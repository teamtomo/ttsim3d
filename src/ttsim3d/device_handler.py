"""Handles cpu/gpu device selection."""

from typing import Optional, Union

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


def get_device(gpu_ids: Optional[Union[int, list[int]]] = None) -> torch.device:
    """Get the appropriate torch device based on availability and user preference.

    Parameters
    ----------
    gpu_ids : Optional[Union[int, list[int]]]
        Device selection preference:
        - None: Use CPU
        - -1: Use first available GPU (CUDA or MPS)
        - >=0: Use specific CUDA device
        - list[int]: Use specific CUDA devices (for multi-GPU)

    Returns
    -------
    torch.device
        The selected compute device
    """
    # Default to CPU
    if gpu_ids is None:
        return torch.device("cpu")

    # Check for CUDA availability
    if torch.cuda.is_available():
        if isinstance(gpu_ids, list):
            # Multi-GPU not yet implemented
            return torch.device(f"cuda:{gpu_ids[0]}")
        elif gpu_ids >= 0:
            return torch.device(f"cuda:{gpu_ids}")
        else:  # gpu_ids == -1
            return torch.device("cuda:0")

    # Check for MPS (Apple Silicon) availability
    elif torch.backends.mps.is_available():
        if gpu_ids is not None:  # User requested GPU
            return torch.device("mps")

    # Fallback to CPU
    return torch.device("cpu")


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to the specified device if it's not already there."""
    if tensor.device != device:
        return tensor.to(device)
    return tensor


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
