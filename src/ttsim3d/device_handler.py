"""Handles cpu/gpu device selection."""

from typing import Optional

import torch


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
