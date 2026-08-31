"""Calculate the scattering potential using monorepo Peng parameter tables."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import einops
import gemmi
import torch
from torch_calculate_electrostatic_potential import BondedScatteringFactorTable
from torch_calculate_electrostatic_potential.utils.peng_model import (
    resolve_scattering_parameters,
)


def _atomic_numbers(atom_ids: list[str]) -> torch.Tensor:
    return torch.tensor(
        [gemmi.Element(atom_id).atomic_number for atom_id in atom_ids],
        dtype=torch.int64,
    )


def _bonded_table_from_dicts(
    params_a: Mapping[str, torch.Tensor | list[float]] | None,
    params_b: Mapping[str, torch.Tensor | list[float]] | None,
) -> BondedScatteringFactorTable | None:
    if params_a is None or params_b is None:
        return None

    def _as_float_list(value: torch.Tensor | list[float]) -> list[float]:
        if isinstance(value, torch.Tensor):
            return [float(item) for item in value.detach().cpu().tolist()]
        return [float(item) for item in value]

    return BondedScatteringFactorTable(
        parameters_a={key: _as_float_list(value) for key, value in params_a.items()},
        parameters_b={key: _as_float_list(value) for key, value in params_b.items()},
    )


def _custom_bonded_providers(
    params_a_protein: Mapping[str, torch.Tensor] | None,
    params_b_protein: Mapping[str, torch.Tensor] | None,
    params_a_rna: Mapping[str, torch.Tensor] | None,
    params_b_rna: Mapping[str, torch.Tensor] | None,
) -> dict[str, BondedScatteringFactorTable] | None:
    protein = _bonded_table_from_dicts(params_a_protein, params_b_protein)
    rna = _bonded_table_from_dicts(params_a_rna, params_b_rna)
    if protein is None and rna is None:
        return None
    providers: dict[str, BondedScatteringFactorTable] = {}
    if protein is not None:
        providers["protein"] = protein
    if rna is not None:
        providers["rna"] = rna
    return providers


def _apply_parameter_overrides(
    parameters: torch.Tensor,
    bonded_ids: list[str],
    molecule_types: list[str],
    overrides_by_molecule: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    result = parameters.clone()
    for index, (bonded_id, molecule_type) in enumerate(
        zip(bonded_ids, molecule_types, strict=True)
    ):
        overrides = overrides_by_molecule.get(molecule_type.strip().lower())
        if overrides is not None and bonded_id in overrides:
            result[index] = overrides[bonded_id].to(
                device=result.device, dtype=result.dtype
            )
    return result


def resolve_atom_scattering_parameters(
    atom_ids: list[str],
    *,
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
    params_a_protein: dict[str, torch.Tensor] | None = None,
    params_a_rna: dict[str, torch.Tensor] | None = None,
    params_b_protein: dict[str, torch.Tensor] | None = None,
    params_b_rna: dict[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-atom Peng ``a`` and ``b`` parameters for the splatting kernel."""
    atomic_numbers = _atomic_numbers(atom_ids)
    use_bonded = atom_bonded_ids is not None and molecule_type is not None

    if not use_bonded:
        return resolve_scattering_parameters(
            atomic_numbers,
            scattering_factors="peng_elemental",
            device=device,
            dtype=dtype,
        )

    custom_providers = _custom_bonded_providers(
        params_a_protein,
        params_b_protein,
        params_a_rna,
        params_b_rna,
    )
    scattering_factors: str | dict[str, BondedScatteringFactorTable] = (
        custom_providers if custom_providers is not None else "peng_bonded"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        params_a, params_b = resolve_scattering_parameters(
            atomic_numbers,
            scattering_factors=scattering_factors,
            bonded_environments=tuple(atom_bonded_ids),
            molecule_types=tuple(molecule_type),
            bonded_fallback="elemental",
            device=device,
            dtype=dtype,
        )

    if params_a_protein:
        params_a = _apply_parameter_overrides(
            params_a,
            atom_bonded_ids,
            molecule_type,
            {"protein": params_a_protein},
        )
    if params_a_rna:
        params_a = _apply_parameter_overrides(
            params_a, atom_bonded_ids, molecule_type, {"rna": params_a_rna}
        )
    if params_b_protein:
        params_b = _apply_parameter_overrides(
            params_b,
            atom_bonded_ids,
            molecule_type,
            {"protein": params_b_protein},
        )
    if params_b_rna:
        params_b = _apply_parameter_overrides(
            params_b, atom_bonded_ids, molecule_type, {"rna": params_b_rna}
        )

    return params_a, params_b


def get_a_param(
    atom_ids: list[str],
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
    params_a_protein: dict[str, torch.Tensor] | None = None,
    params_a_rna: dict[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Get the ``a`` scattering parameters for a set of atoms."""
    params_a, _ = resolve_atom_scattering_parameters(
        atom_ids,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
        params_a_protein=params_a_protein,
        params_a_rna=params_a_rna,
        device=device,
    )
    return params_a


def get_b_param(
    atom_ids: list[str],
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
    params_b_protein: dict[str, torch.Tensor] | None = None,
    params_b_rna: dict[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Get the ``b`` scattering parameters for a set of atoms."""
    _, params_b = resolve_atom_scattering_parameters(
        atom_ids,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
        params_b_protein=params_b_protein,
        params_b_rna=params_b_rna,
        device=device,
    )
    return params_b


def get_total_b_param(
    atom_ids: list[str],
    atom_b_factors: torch.Tensor,
    atom_bonded_ids: list[str] | None = None,
    molecule_type: list[str] | None = None,
    params_b_protein: dict[str, torch.Tensor] | None = None,
    params_b_rna: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Calculate the total B parameter per atom."""
    b_params = get_b_param(
        atom_ids,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
        params_b_protein=params_b_protein,
        params_b_rna=params_b_rna,
        device=atom_b_factors.device,
    ).to(atom_b_factors.device)
    return 2 * torch.pi / torch.sqrt(atom_b_factors.unsqueeze(1) + b_params)


def get_scattering_potential_of_voxel_batch(
    zyx_coords1: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    zyx_coords2: torch.Tensor,  # Shape: (atomN, voxelN, 3)
    atom_ids: list[str],  # Shape: (atomN)
    atom_b_factors: torch.Tensor,  # Shape: (atomN)
    lead_term: float,
    device: torch.device = None,
    atom_bonded_ids: list[str] | None = None,  # Shape: (atomN)
    molecule_type: list[str] | None = None,
    params_a_protein: dict[str, torch.Tensor] | None = None,
    params_a_rna: dict[str, torch.Tensor] | None = None,
    params_b_protein: dict[str, torch.Tensor] | None = None,
    params_b_rna: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Batched calculation over atoms for scattering potential at each voxel."""
    if device is None:
        device = zyx_coords1.device

    params_a, params_b = resolve_atom_scattering_parameters(
        atom_ids,
        atom_bonded_ids=atom_bonded_ids,
        molecule_type=molecule_type,
        params_a_protein=params_a_protein,
        params_a_rna=params_a_rna,
        params_b_protein=params_b_protein,
        params_b_rna=params_b_rna,
        device=device,
    )
    params_bPlusB = 2 * torch.pi / torch.sqrt(
        atom_b_factors.unsqueeze(1).to(device=device, dtype=params_b.dtype) + params_b
    )

    t_all = (zyx_coords1 * zyx_coords2) >= 0
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
    temp_potential = einops.reduce(torch.abs(t0) * params_a, "a n i -> a n", "sum")

    return (lead_term * temp_potential).float()
