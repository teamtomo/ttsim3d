"""Load atomic structures for simulation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import mmdf
from torch_structure_manipulation import AtomicStructure

if TYPE_CHECKING:
    import os

    import torch


def load_structure(
    path: str | os.PathLike[str],
    *,
    center: bool = True,
    remove_hydrogens: bool = True,
    annotate_bonding: bool = False,
    device: torch.device | str | None = None,
) -> AtomicStructure:
    """Read a PDB or mmCIF file into an ``AtomicStructure``.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a PDB or mmCIF file readable by ``mmdf``.
    center : bool
        If True (default), translate the atoms so their mean position is at the
        origin. Set to False to keep the coordinates of a pre-aligned model.
    remove_hydrogens : bool
        If True (default), drop hydrogen atoms.
    annotate_bonding : bool
        If True, annotate per-atom bonding environments and molecule types, as
        required by ``scattering_factors="peng_bonded"``. Annotation runs before
        hydrogen removal so environment keys are independent of
        ``remove_hydrogens``.
    device : torch.device | str | None
        Device for the returned tensors. Default is the CPU.

    Returns
    -------
    AtomicStructure
        Positions (Angstroms, ZYX order), atomic numbers, B-factors (Angstroms
        squared), occupancies, and optional bonding metadata.
    """
    df = mmdf.read(path)

    if annotate_bonding:
        structure = AtomicStructure.from_annotated_dataframe(
            df, include_hydrogens=True, device=device
        )
    else:
        structure = AtomicStructure.from_dataframe(df, device=device)

    if remove_hydrogens:
        structure = _select_atoms(structure, structure.atomic_numbers != 1)

    if structure.num_atoms == 0:
        raise ValueError(f"No atoms left to simulate in structure {path}.")

    if center:
        positions = structure.positions_zyx
        centered = positions - positions.mean(dim=-2, keepdim=True)
        structure = structure.with_positions(centered)

    return structure


def adjust_b_factors(
    structure: AtomicStructure,
    scale: float = 1.0,
    offset: float = 0.0,
) -> AtomicStructure:
    """Return a copy of ``structure`` with B-factors ``scale * B + offset``.

    Parameters
    ----------
    structure : AtomicStructure
        Structure whose B-factors to adjust.
    scale : float
        Multiplicative factor applied to every B-factor. Default is 1.0.
    offset : float
        Additive B-factor, in Angstroms squared, applied after scaling. Default
        is 0.0.

    Returns
    -------
    AtomicStructure
        The structure with adjusted B-factors.
    """
    if scale == 1.0 and offset == 0.0:
        return structure
    b_factors = structure.b_factors * scale + offset
    return dataclasses.replace(structure, b_factors=b_factors)


def _select_atoms(structure: AtomicStructure, mask: torch.Tensor) -> AtomicStructure:
    """Keep the atoms where the boolean ``mask`` (shape ``(n_atoms,)``) is True."""
    if mask.ndim != 1:
        raise ValueError("Atom selection requires one-dimensional atomic numbers.")
    keep = mask.cpu().tolist()

    def _select_text(values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        return tuple(value for value, k in zip(values, keep, strict=True) if k)

    def _select_numeric(values: torch.Tensor) -> torch.Tensor:
        return values if values.ndim == 0 else values[..., mask]

    return AtomicStructure(
        positions_zyx=structure.positions_zyx[..., mask, :],
        atomic_numbers=structure.atomic_numbers[..., mask],
        elements=_select_text(structure.elements) or (),
        atom_names=_select_text(structure.atom_names) or (),
        b_factors=_select_numeric(structure.b_factors),
        occupancies=_select_numeric(structure.occupancies),
        bonded_environments=_select_text(structure.bonded_environments),
        molecule_types=_select_text(structure.molecule_types),
    )
