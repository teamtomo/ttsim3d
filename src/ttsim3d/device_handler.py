"""Handles cpu/gpu device selection."""

import multiprocessing as mp
from typing import Optional

import torch


def get_cpu_cores() -> int:
    """
    Get the number of CPU cores available.

    Returns
    -------
        int: Number of CPU cores available.
    """
    return mp.cpu_count()


def select_gpus(
    gpu_ids: Optional[list[int]] = None,
    num_gpus: int = 1,
) -> list[torch.device]:
    """
    Select multiple GPU devices based on IDs or available memory.

    Args:
        gpu_ids: List of specific GPU IDs to use.
                If None, selects GPUs with most available memory.
        num_gpus: Number of GPUs to use if gpu_ids is None.

    Returns
    -------
        list[torch.device]: Selected GPU devices or [CPU] if no GPU available
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
    if gpu_ids is not None:
        valid_devices = []
        for gpu_id in gpu_ids:
            if gpu_id >= n_gpus:
                print(f"Requested GPU {gpu_id} not available. Max GPU ID is {n_gpus-1}")
                continue
            valid_devices.append(torch.device(f"cuda:{gpu_id}"))

        if not valid_devices:
            print("No valid GPUs specified. Using CPU")
            return [torch.device("cpu")]
        return valid_devices

    # Find GPUs with most available memory
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

    # Sort by available memory and select the top num_gpus
    gpu_memory_available.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [
        torch.device(f"cuda:{idx}") for idx, _ in gpu_memory_available[:num_gpus]
    ]

    print("\nSelected GPUs:", [str(device) for device in selected_gpus])
    return selected_gpus


def calculate_batch_size_gpu(
    total_atoms: int,
    neighborhood_size: int,
    device: torch.device,
    safety_factor: float = 0.8,  # Use only 80% of available memory by default
    min_batch_size: int = 100,
) -> int:
    """
    Calculate optimal batch size based on available GPU memory and data size.

    Args:
        total_atoms: Total number of atoms to process
        neighborhood_size: Size of neighborhood around each atom
        device: PyTorch device (GPU)
        safety_factor: Fraction of available memory to use (0.0 to 1.0)

    Returns
    -------
        Optimal batch size
    """
    # Get available GPU memory in bytes
    gpu_memory = torch.cuda.get_device_properties(device).total_memory

    # Calculate memory requirements per atom
    voxels_per_atom = (2 * neighborhood_size + 1) ** 3
    bytes_per_float = 4  # 32-bit float

    # Memory needed for:
    # 1. Voxel positions (float32): batch_size * voxels_per_atom * 3 coordinates
    # 2. Valid mask (bool): batch_size * voxels_per_atom
    # 3. Relative coordinates (float32): batch_size * voxels_per_atom * 3
    # 4. Potentials (float32): batch_size * voxels_per_atom
    # Plus some overhead for temporary variables
    memory_per_atom = (
        voxels_per_atom * (3 * bytes_per_float)  # Voxel positions
        + voxels_per_atom * 1  # Valid mask (bool)
        + voxels_per_atom * (3 * bytes_per_float)  # Relative coordinates
        + voxels_per_atom * bytes_per_float  # Potentials
        + 1024  # Additional overhead
    )

    # Calculate batch size
    optimal_batch_size = int((gpu_memory * safety_factor) / memory_per_atom)

    # Ensure batch size is at least 1 but not larger than total atoms
    optimal_batch_size = max(min_batch_size, min(optimal_batch_size, total_atoms))

    print(f"Total GPU memory: {gpu_memory / 1024**3:.2f} GB")
    # print(f"Available GPU memory: {gpu_memory_available / 1024**3:.2f} GB")
    print(f"Estimated memory per atom: {memory_per_atom / 1024**2:.2f} MB")
    print(f"Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size
