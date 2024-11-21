"""Calculates the scatttering potential."""

import json
from pathlib import Path

import einops
import torch


def get_scattering_parameters() -> tuple[dict, dict]:
    """
    Load scattering parameters from JSON file.

    Args:
        None

    Returns
    -------
        scattering_params_a: dict
            Scattering parameters for atom type A.
        scattering_params_b: dict
            Scattering parameters for atom type B.
    """
    scattering_param_path = Path(__file__).parent / "elastic_scattering_factors.json"

    with open(scattering_param_path) as f:
        data = json.load(f)

    scattering_params_a = {k: v for k, v in data["parameters_a"].items() if v != []}
    scattering_params_b = {k: v for k, v in data["parameters_b"].items() if v != []}
    return scattering_params_a, scattering_params_b


def get_a_param(
    scattering_params_a: dict,
    atoms_id_filtered: list[str],
) -> torch.Tensor:
    """
    Get the 'a' scattering parameters.

    Args:
        scattering_params_a: dict
            Scattering parameters for atom type A.
        atoms_id_filtered: list[str]
            Atom IDs.

    Returns
    -------
        params_tensor: torch.Tensor
            Scattering parameters for each atom in the neighborhood
    """
    # Iterate over each atom_id in atoms_id_filtered
    params_list = [scattering_params_a[atom_id] for atom_id in atoms_id_filtered]
    # Convert the list to a PyTorch tensor with the desired shape
    params_tensor = torch.tensor(params_list)
    return params_tensor


def get_total_b_param(
    scattering_params_b: dict,
    atoms_id_filtered: list[str],
    atoms_b_factor_scaled_filtered: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the total B parameter for each atom in the neighborhood.

    Args:
        scattering_params_b: dict
            Scattering parameters for atom type B.
        atoms_id_filtered: list[str]
            Atom IDs.
        atoms_b_factor_scaled_filtered: torch.Tensor
            Atom B factors.

    Returns
    -------
        bPlusB: torch.Tensor
            Total B parameter for each atom in the neighborhood.
    """
    b_params = torch.stack(
        [torch.tensor(scattering_params_b[atom_id]) for atom_id in atoms_id_filtered]
    )
    bPlusB = (
        2
        * torch.pi
        / torch.sqrt(atoms_b_factor_scaled_filtered.unsqueeze(1) + b_params)
    )
    return bPlusB


def get_scattering_potential_of_voxel(
    zyx_coords1: torch.Tensor,  # Shape: (N, 3)
    zyx_coords2: torch.Tensor,  # Shape: (N, 3)
    bPlusB: torch.Tensor,  # Shape: (5)
    lead_term: float,
    scattering_params_a: torch.Tensor,  # Shape: (5)
    device: torch.device = None,
) -> torch.Tensor:
    """
    Calculate scattering potential for all voxels in the neighborhood of of the atom.

    Args:
        zyx_coords1: torch.Tensor
            Coordinates of the first voxel in the neighborhood.
        zyx_coords2: torch.Tensor
            Coordinates of the second voxel in the neighborhood.
        bPlusB: torch.Tensor
            Total B parameter for each atom in the neighborhood.
        atom_id: str
            Atom ID.
        lead_term: float
            Lead term for the scattering potential.
        scattering_params_a: torch.Tensor
            Scattering parameters for atom type A.
        device: torch.device
            Device to run the computation on.

    Returns
    -------
        potential: torch.Tensor
            Scattering potential for all voxels in the neighborhood.
    """
    # If device not specified, use the device of input tensors
    if device is None:
        device = zyx_coords1.device

    # Compare signs element-wise for batched coordinates
    t_all = (zyx_coords1 * zyx_coords2) >= 0  # Shape: (N, 3)
    temp_potential = torch.zeros(len(zyx_coords1), device=device)

    # rearrange for broadcasting
    zyx_coords1 = einops.rearrange(zyx_coords1, "n d -> n d 1")
    zyx_coords2 = einops.rearrange(zyx_coords2, "n d -> n d 1")
    t_all = einops.rearrange(t_all, "n d -> n d 1")
    bPlusB = einops.rearrange(bPlusB, "i -> 1 1 i")

    all_terms = torch.where(
        t_all,
        torch.special.erf(bPlusB * zyx_coords2)
        - torch.special.erf(bPlusB * zyx_coords1),
        torch.abs(torch.special.erf(bPlusB * zyx_coords2))
        + torch.abs(torch.special.erf(bPlusB * zyx_coords1)),
    )
    t0 = einops.reduce(all_terms, "n d i-> n i", "prod")
    a_mult = torch.abs(t0) * scattering_params_a
    temp_potential = einops.reduce(a_mult, "n i -> n", "sum")
    return lead_term * temp_potential


def get_scattering_potential_of_voxel_batch(
    zyx_coords1: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    zyx_coords2: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    bPlusB: torch.Tensor,  # Shape: (atomN, 5)
    lead_term: float,
    scattering_params_a: torch.Tensor,  # Shape: (atomN, 5)
    device: torch.device = None,
) -> torch.Tensor:
    """
    Calculate scattering potential for all voxels in the neighborhood of of the atom.

    Args:
        zyx_coords1: torch.Tensor
            Coordinates of the first voxel in the neighborhood.
        zyx_coords2: torch.Tensor
            Coordinates of the second voxel in the neighborhood.
        bPlusB: torch.Tensor
            Total B parameter for each atom in the neighborhood.
        atom_id: str
            Atom ID.
        lead_term: float
            Lead term for the scattering potential.
        scattering_params_a: torch.Tensor
            Scattering parameters for atom type A.
        device: torch.device
            Device to run the computation on.

    Returns
    -------
        potential: torch.Tensor
            Scattering potential for all voxels in the neighborhood.
    """
    # If device not specified, use the device of input tensors
    if device is None:
        device = zyx_coords1.device

    # Compare signs element-wise for batched coordinates
    t_all = (zyx_coords1 * zyx_coords2) >= 0  # Shape: (atomN, voxelN, 3)
    temp_potential = torch.zeros(len(zyx_coords1), device=device)

    # rearrange for broadcasting
    zyx_coords1 = einops.rearrange(zyx_coords1, "a n d -> a n d 1")
    zyx_coords2 = einops.rearrange(zyx_coords2, "a n d -> a n d 1")
    t_all = einops.rearrange(t_all, "a n d -> a n d 1")
    bPlusB = einops.rearrange(bPlusB, "a i -> a 1 1 i")

    all_terms = torch.where(
        t_all,
        torch.special.erf(bPlusB * zyx_coords2)
        - torch.special.erf(bPlusB * zyx_coords1),
        torch.abs(torch.special.erf(bPlusB * zyx_coords2))
        + torch.abs(torch.special.erf(bPlusB * zyx_coords1)),
    )
    t0 = einops.reduce(all_terms, "a n d i -> a n i", "prod")
    scattering_params_a = einops.rearrange(scattering_params_a, "a i -> a 1 i")
    a_mult = torch.abs(t0) * scattering_params_a
    temp_potential = einops.reduce(a_mult, "a n i -> a n", "sum")
    return (lead_term * temp_potential).float()
