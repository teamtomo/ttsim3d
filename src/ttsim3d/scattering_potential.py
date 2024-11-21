"""Calculates the scatttering potential."""

import json
from pathlib import Path

import einops
import torch


def get_scattering_parameters() -> tuple[dict, dict]:
    """Load scattering parameters from JSON file.

    NOTE: This function currently assumes the JSON file path relative to this
    file in the package. Could add future functionality to specify the path.

    Parameters
    ----------
    None

    Returns
    -------
    scattering_params_a : dict
        Scattering parameters for atom type A.
    scattering_params_b : dict
        Scattering parameters for atom type B.
    """
    scattering_param_path = Path(__file__).parent / "elastic_scattering_factors.json"

    with open(scattering_param_path) as f:
        data = json.load(f)

    scattering_params_a = {k: v for k, v in data["parameters_a"].items() if v != []}
    scattering_params_b = {k: v for k, v in data["parameters_b"].items() if v != []}

    return scattering_params_a, scattering_params_b


SCATTERING_PARAMS_A, SCATTERING_PARAMS_B = get_scattering_parameters()


def get_a_param(atom_ids: list[str]) -> torch.Tensor:
    """Get the 'a' scattering parameters for a set of atoms.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.

    Returns
    -------
    params_tensor: torch.Tensor
        Scattering parameters for each atom in the neighborhood
    """
    # Iterate over each atom_id in atoms_id_filtered
    params_list = [SCATTERING_PARAMS_A[atom_id] for atom_id in atom_ids]
    params_tensor = torch.tensor(params_list)

    return params_tensor


def get_b_param(atom_ids: list[str]) -> torch.Tensor:
    """Get the 'b' scattering parameters for a set of atoms.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.

    Returns
    -------
    params_tensor: torch.Tensor
        Scattering parameters for each atom in the neighborhood
    """
    # Iterate over each atom_id in atoms_id_filtered
    params_list = [SCATTERING_PARAMS_B[atom_id] for atom_id in atom_ids]
    params_tensor = torch.tensor(params_list)

    return params_tensor


def get_total_b_param(
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
) -> torch.Tensor:
    """Calculate the total B parameter per atom.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.
    atom_b_factors : torch.Tensor
        Atom B factors.

    Returns
    -------
    bPlusB: torch.Tensor
        Total B parameter for each atom in the neighborhood.
    """
    b_params = get_b_param(atom_ids)
    bPlusB = 2 * torch.pi / torch.sqrt(atom_b_factors.unsqueeze(1) + b_params)

    return bPlusB


# def get_scattering_potential_of_voxel(
#     zyx_coords1: torch.Tensor,  # Shape: (N, 3)
#     zyx_coords2: torch.Tensor,  # Shape: (N, 3)
#     bPlusB: torch.Tensor,  # Shape: (5)
#     lead_term: float,
#     scattering_params_a: torch.Tensor,  # Shape: (5)
#     device: torch.device = None,
# ) -> torch.Tensor:
#     """
#     Calculate scattering potential for all voxels in the neighborhood of of the atom.

#     Args:
#         zyx_coords1: torch.Tensor
#             Coordinates of the first voxel in the neighborhood.
#         zyx_coords2: torch.Tensor
#             Coordinates of the second voxel in the neighborhood.
#         bPlusB: torch.Tensor
#             Total B parameter for each atom in the neighborhood.
#         atom_id: str
#             Atom ID.
#         lead_term: float
#             Lead term for the scattering potential.
#         scattering_params_a: torch.Tensor
#             Scattering parameters for atom type A.
#         device: torch.device
#             Device to run the computation on.

#     Returns
#     -------
#         potential: torch.Tensor
#             Scattering potential for all voxels in the neighborhood.
#     """
#     # If device not specified, use the device of input tensors
#     if device is None:
#         device = zyx_coords1.device

#     # Compare signs element-wise for batched coordinates
#     t_all = (zyx_coords1 * zyx_coords2) >= 0  # Shape: (N, 3)
#     temp_potential = torch.zeros(len(zyx_coords1), device=device)

#     # rearrange for broadcasting
#     zyx_coords1 = einops.rearrange(zyx_coords1, "n d -> n d 1")
#     zyx_coords2 = einops.rearrange(zyx_coords2, "n d -> n d 1")
#     t_all = einops.rearrange(t_all, "n d -> n d 1")
#     bPlusB = einops.rearrange(bPlusB, "i -> 1 1 i")

#     all_terms = torch.where(
#         t_all,
#         torch.special.erf(bPlusB * zyx_coords2)
#         - torch.special.erf(bPlusB * zyx_coords1),
#         torch.abs(torch.special.erf(bPlusB * zyx_coords2))
#         + torch.abs(torch.special.erf(bPlusB * zyx_coords1)),
#     )
#     t0 = einops.reduce(all_terms, "n d i-> n i", "prod")
#     a_mult = torch.abs(t0) * scattering_params_a
#     temp_potential = einops.reduce(a_mult, "n i -> n", "sum")
#     return lead_term * temp_potential


def get_scattering_potential_of_voxel_batch(
    zyx_coords1: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    zyx_coords2: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    atom_ids: list[str],  # Shape: (atomN)
    atom_b_factors: torch.Tensor,  # Shape: (atomN)
    lead_term: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Batched calculation over atoms for scattering potential at each voxel.

    Follows the equation 12 from https://journals.iucr.org/m/issues/2021/06/00/rq5007/index.html
    for calculating the scattering potential per voxel.

    Parameters
    ----------
    zyx_coords1 : torch.Tensor
        Coordinates of the first voxel in the neighborhood. Shape of
        (atomN, voxelN, 3) where atomN is the number of atoms and voxelN
        is the number of voxels in the neighborhood.
    zyx_coords2 : torch.Tensor
        Coordinates of the second voxel in the neighborhood. Shape of
        (atomN, voxelN, 3) where atomN is the number of atoms and voxelN
        is the number of voxels in the neighborhood.
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.
    atom_b_factors : torch.Tensor
        Atom B factors.
    lead_term : float
        Lead term for the scattering potential.
    device: torch.device
        Device to run the computation on.

    Returns
    -------
    potential : torch.Tensor
        Scattering potential for all voxels in the neighborhood.
    """
    # a   -> atom index with shape (n_atoms)
    # d   -> dimension with shape (3)
    # v/n -> flattened voxel indices for neighborhood with shape (h*w*d)
    # i   -> scattering fit index with shape (5)

    # If device not specified, use the device of input tensors
    if device is None:
        device = zyx_coords1.device

    # Get scattering parameters for atoms
    params_a = get_a_param(atom_ids)
    params_bPlusB = get_total_b_param(atom_ids, atom_b_factors)

    # Compare signs element-wise for batched coordinates
    t_all = (zyx_coords1 * zyx_coords2) >= 0  # Shape: (atomN, voxelN, 3)
    temp_potential = torch.zeros(len(zyx_coords1), device=device)

    # rearrange for broadcasting
    zyx_coords1 = einops.rearrange(zyx_coords1, "a n d -> a n d 1")
    zyx_coords2 = einops.rearrange(zyx_coords2, "a n d -> a n d 1")
    t_all = einops.rearrange(t_all, "a n d -> a n d 1")
    params_bPlusB = einops.rearrange(params_bPlusB, "a i -> a 1 1 i")

    all_terms = torch.where(
        t_all,
        torch.special.erf(params_bPlusB * zyx_coords2)
        - torch.special.erf(params_bPlusB * zyx_coords1),
        torch.abs(torch.special.erf(params_bPlusB * zyx_coords2))
        + torch.abs(torch.special.erf(params_bPlusB * zyx_coords1)),
    )
    t0 = einops.reduce(all_terms, "a n d i -> a n i", "prod")
    params_a = einops.rearrange(params_a, "a i -> a 1 i")
    a_mult = torch.abs(t0) * params_a
    temp_potential = einops.reduce(a_mult, "a n i -> a n", "sum")

    return (lead_term * temp_potential).float()
