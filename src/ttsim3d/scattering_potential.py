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


def get_bonded_scattering_parameters_protein() -> tuple[dict, dict]:
    """Load bonded scattering parameters for protein from JSON file.

    Returns
    -------
    scattering_params_a : dict
        Bonded scattering parameters for atom type A (protein).
    scattering_params_b : dict
        Bonded scattering parameters for atom type B (protein).
    """
    scattering_param_path = (
        Path(__file__).parent / "elastic_scattering_bonding_protein.json"
    )

    with open(scattering_param_path) as f:
        data = json.load(f)

    scattering_params_a = {k: v for k, v in data["parameters_a"].items() if v != []}
    scattering_params_b = {k: v for k, v in data["parameters_b"].items() if v != []}

    return scattering_params_a, scattering_params_b


def get_bonded_scattering_parameters_rna() -> tuple[dict, dict]:
    """Load bonded scattering parameters for RNA from JSON file.

    Returns
    -------
    scattering_params_a : dict
        Bonded scattering parameters for atom type A (RNA).
    scattering_params_b : dict
        Bonded scattering parameters for atom type B (RNA).
    """
    scattering_param_path = (
        Path(__file__).parent / "elastic_scattering_bonding_rna.json"
    )

    with open(scattering_param_path) as f:
        data = json.load(f)

    scattering_params_a = {k: v for k, v in data["parameters_a"].items() if v != []}
    scattering_params_b = {k: v for k, v in data["parameters_b"].items() if v != []}

    return scattering_params_a, scattering_params_b


SCATTERING_PARAMS_A, SCATTERING_PARAMS_B = get_scattering_parameters()
SCATTERING_PARAMS_A_PROTEIN, SCATTERING_PARAMS_B_PROTEIN = (
    get_bonded_scattering_parameters_protein()
)
SCATTERING_PARAMS_A_RNA, SCATTERING_PARAMS_B_RNA = (
    get_bonded_scattering_parameters_rna()
)


def get_a_param(
    atom_ids: list[str],
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
) -> torch.Tensor:
    """Get the 'a' scattering parameters for a set of atoms.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.
    atom_bonded_ids : list[str] | None
        Bonded atom IDs (e.g., "C(HHCN)") for each atom. If None, uses standard
        scattering factors.
    molecule_type : list[str] | None
        Molecule types (e.g., ["protein", "rna"]). Used to select which bonded
        scattering factors to use.

    Returns
    -------
    params_tensor: torch.Tensor
        Scattering parameters for each atom in the neighborhood
    """
    # Use bonded scattering factors if available
    if atom_bonded_ids is not None and molecule_type is not None:
        params_list = []
        for i, bonded_id in enumerate(atom_bonded_ids):
            # Determine which params to use based on molecule type
            # Try protein params if "protein" in molecule_type
            if "protein" in molecule_type and bonded_id in SCATTERING_PARAMS_A_PROTEIN:
                params_list.append(SCATTERING_PARAMS_A_PROTEIN[bonded_id])
            # Try RNA params if "rna" in molecule_type
            elif "rna" in molecule_type and bonded_id in SCATTERING_PARAMS_A_RNA:
                params_list.append(SCATTERING_PARAMS_A_RNA[bonded_id])
            # Fallback to standard atom_id if bonded_id not found in appropriate dict
            else:
                params_list.append(SCATTERING_PARAMS_A[atom_ids[i]])
    else:
        # Use standard scattering factors
        params_list = [SCATTERING_PARAMS_A[atom_id] for atom_id in atom_ids]

    params_tensor = torch.tensor(params_list)
    return params_tensor


def get_b_param(
    atom_ids: list[str],
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
) -> torch.Tensor:
    """Get the 'b' scattering parameters for a set of atoms.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.
    atom_bonded_ids : list[str] | None
        Bonded atom IDs (e.g., "C(HHCN)") for each atom. If None, uses standard
        scattering factors.
    molecule_type : list[str] | None
        Molecule types (e.g., ["protein", "rna"]). Used to select which bonded
        scattering factors to use.

    Returns
    -------
    params_tensor: torch.Tensor
        Scattering parameters for each atom in the neighborhood
    """
    # Use bonded scattering factors if available
    if atom_bonded_ids is not None and molecule_type is not None:
        params_list = []
        for i, bonded_id in enumerate(atom_bonded_ids):
            # Determine which params to use based on molecule type
            # Try protein params if "protein" in molecule_type
            if "protein" in molecule_type and bonded_id in SCATTERING_PARAMS_B_PROTEIN:
                params_list.append(SCATTERING_PARAMS_B_PROTEIN[bonded_id])
            # Try RNA params if "rna" in molecule_type
            elif "rna" in molecule_type and bonded_id in SCATTERING_PARAMS_B_RNA:
                params_list.append(SCATTERING_PARAMS_B_RNA[bonded_id])
            # Fallback to standard atom_id if bonded_id not found in appropriate dict
            else:
                params_list.append(SCATTERING_PARAMS_B[atom_ids[i]])
    else:
        # Use standard scattering factors
        params_list = [SCATTERING_PARAMS_B[atom_id] for atom_id in atom_ids]

    params_tensor = torch.tensor(params_list)
    return params_tensor


def get_total_b_param(
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
) -> torch.Tensor:
    """Calculate the total B parameter per atom.

    Parameters
    ----------
    atom_ids : list[str]
        Atom IDs as a list of uppercase element symbols.
    atom_b_factors : torch.Tensor
        Atom B factors.
    atom_bonded_ids : list[str] | None
        Bonded atom IDs (e.g., "C(HHCN)") for each atom. If None, uses standard
        scattering factors.
    molecule_type : list[str] | None
        Molecule types (e.g., ["protein", "rna"]). Used to select which bonded
        scattering factors to use.

    Returns
    -------
    bPlusB: torch.Tensor
        Total B parameter for each atom in the neighborhood.
    """
    b_params = get_b_param(
        atom_ids, atom_bonded_ids=atom_bonded_ids, molecule_type=molecule_type
    ).to(atom_b_factors.device)
    bPlusB = 2 * torch.pi / torch.sqrt(atom_b_factors.unsqueeze(1) + b_params)

    return bPlusB


def get_scattering_potential_of_voxel_batch(
    zyx_coords1: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    zyx_coords2: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    atom_ids: list[str],  # Shape: (atomN)
    atom_b_factors: torch.Tensor,  # Shape: (atomN)
    lead_term: float,
    device: torch.device = None,
    atom_bonded_ids: list[str] | None = None,  # Shape: (atomN)
    molecule_type: list[str] | None = None,
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
    atom_bonded_ids : list[str] | None
        Bonded atom IDs (e.g., "C(HHCN)") for each atom. If None, uses standard
        scattering factors.
    molecule_type : list[str] | None
        Molecule types (e.g., ["protein", "rna"]). Used to select which bonded
        scattering factors to use.

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
    params_a = get_a_param(
        atom_ids, atom_bonded_ids=atom_bonded_ids, molecule_type=molecule_type
    ).to(device)
    params_bPlusB = get_total_b_param(
        atom_ids,
        atom_b_factors,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
    ).to(device)

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
