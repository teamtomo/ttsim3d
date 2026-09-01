"""TeamTomo monorepo potential backend for ttsim3d."""

from __future__ import annotations

import warnings

import gemmi
import torch
from torch_calculate_electrostatic_potential import (
    GridConfig,
    default_sublattice_radius,
    potential_from_structure_3d,
)
from torch_calculate_electrostatic_potential.utils.peng_model import ScatteringFactors
from torch_structure_manipulation import AtomicStructure

from ttsim3d.scattering_potential import resolve_scattering_factors_spec


def build_atomic_structure(
    atom_positions_zyx: torch.Tensor,
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    *,
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
) -> AtomicStructure:
    """Build an ``AtomicStructure`` from ttsim3d simulation inputs."""
    atomic_numbers = torch.tensor(
        [gemmi.Element(atom_id).atomic_number for atom_id in atom_ids],
        dtype=torch.int64,
        device=atom_positions_zyx.device,
    )
    return AtomicStructure(
        positions_zyx=atom_positions_zyx,
        atomic_numbers=atomic_numbers,
        elements=tuple(atom_ids),
        atom_names=tuple(atom_ids),
        b_factors=atom_b_factors,
        occupancies=torch.ones_like(atom_b_factors),
        bonded_environments=tuple(atom_bonded_ids) if atom_bonded_ids is not None else None,
        molecule_types=tuple(molecule_type) if molecule_type is not None else None,
    )


def simulate_potential_teamtomo(
    atom_positions_zyx: torch.Tensor,
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    sim_pixel_spacing: float,
    sim_volume_shape: tuple[int, int, int],
    *,
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
    params_a_protein: dict[str, torch.Tensor] | None = None,
    params_a_rna: dict[str, torch.Tensor] | None = None,
    params_b_protein: dict[str, torch.Tensor] | None = None,
    params_b_rna: dict[str, torch.Tensor] | None = None,
    atom_batch_size: int = 4096,
    sublattice_radius: float | None = None,
) -> torch.Tensor:
    """Calculate a 3D potential volume using ``potential_from_structure_3d``."""
    device = atom_positions_zyx.device
    structure = build_atomic_structure(
        atom_positions_zyx,
        atom_ids,
        atom_b_factors,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
    )
    radius = (
        default_sublattice_radius(sim_pixel_spacing)
        if sublattice_radius is None
        else sublattice_radius
    )
    grid_config = GridConfig.from_grid_shape_and_voxel_size(
        grid_shape=sim_volume_shape,
        voxel_size=(sim_pixel_spacing, sim_pixel_spacing, sim_pixel_spacing),
        center_zyx=(0.0, 0.0, 0.0),
        sublattice_radius=radius,
        device=device,
    )
    scattering_factors: ScatteringFactors = resolve_scattering_factors_spec(
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
        params_a_protein=params_a_protein,
        params_b_protein=params_b_protein,
        params_a_rna=params_a_rna,
        params_b_rna=params_b_rna,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return potential_from_structure_3d(
            structure,
            grid_config,
            scattering_factors=scattering_factors,
            bonded_fallback="elemental",
            per_voxel_averaging=True,
            batch_size=atom_batch_size,
            verbose=False,
        )
