"""Calculates the scatttering potential."""

import json
from pathlib import Path

import torch
from scipy import constants as C


def calculate_relativistic_electron_wavelength(energy: float) -> float:
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy: float
        acceleration potential in volts.

    Returns
    -------
    wavelength: float
        relativistic wavelength of the electron in meters.
    """
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = energy
    eV = e * V

    numerator = h * c
    denominator = (eV * (2 * m0 * c**2 + eV)) ** 0.5
    return float(numerator / denominator)


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
    bPlusB: torch.Tensor,
    atom_id: str,
    lead_term: float,
    scattering_params_a: dict,  # Add parameter dictionary
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
        scattering_params_a: dict
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

    # Get scattering parameters for this atom type and move to correct device
    # Convert parameters to tensor and move to device
    if isinstance(scattering_params_a[atom_id], torch.Tensor):
        a_params = scattering_params_a[atom_id].clone().detach().to(device)
    else:
        a_params = torch.as_tensor(scattering_params_a[atom_id], device=device)

    # Compare signs element-wise for batched coordinates
    t1 = (zyx_coords1[:, 2] * zyx_coords2[:, 2]) >= 0  # Shape: (N,)
    t2 = (zyx_coords1[:, 1] * zyx_coords2[:, 1]) >= 0  # Shape: (N,)
    t3 = (zyx_coords1[:, 0] * zyx_coords2[:, 0]) >= 0  # Shape: (N,)

    temp_potential = torch.zeros(len(zyx_coords1), device=device)

    for i, bb in enumerate(bPlusB):
        a = a_params[i]
        # Handle x dimension
        x_term = torch.where(
            t1,
            torch.special.erf(bb * zyx_coords2[:, 2])
            - torch.special.erf(bb * zyx_coords1[:, 2]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 2]))
            + torch.abs(torch.special.erf(bb * zyx_coords1[:, 2])),
        )

        # Handle y dimension
        y_term = torch.where(
            t2,
            torch.special.erf(bb * zyx_coords2[:, 1])
            - torch.special.erf(bb * zyx_coords1[:, 1]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 1]))
            + torch.abs(torch.special.erf(bb * zyx_coords1[:, 1])),
        )

        # Handle z dimension
        z_term = torch.where(
            t3,
            torch.special.erf(bb * zyx_coords2[:, 0])
            - torch.special.erf(bb * zyx_coords1[:, 0]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 0]))
            + torch.abs(torch.special.erf(bb * zyx_coords1[:, 0])),
        )

        t0 = z_term * y_term * x_term
        temp_potential += a * torch.abs(t0)

    return lead_term * temp_potential
