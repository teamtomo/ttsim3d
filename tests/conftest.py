"""Shared test fixtures."""

from pathlib import Path

import pytest
import torch
from torch_structure_manipulation import AtomicStructure

# (residue, residue_id, atom name, element, x, y, z) for an ALA-GLY dipeptide.
_DIPEPTIDE_ATOMS = [
    ("ALA", 1, "N", "N", -0.677, -1.230, -0.491),
    ("ALA", 1, "CA", "C", -0.001, 0.064, -0.491),
    ("ALA", 1, "C", "C", 1.499, -0.110, -0.491),
    ("ALA", 1, "O", "O", 2.030, -1.227, -0.502),
    ("ALA", 1, "CB", "C", -0.509, 0.856, 0.727),
    ("ALA", 1, "H", "H", -0.300, -1.900, -1.150),
    ("ALA", 1, "HA", "H", -0.250, 0.620, -1.400),
    ("ALA", 1, "HB1", "H", -1.600, 0.950, 0.700),
    ("ALA", 1, "HB2", "H", -0.100, 1.870, 0.720),
    ("ALA", 1, "HB3", "H", -0.200, 0.360, 1.650),
    ("GLY", 2, "N", "N", 2.205, 1.016, -0.481),
    ("GLY", 2, "CA", "C", 3.660, 1.006, -0.481),
    ("GLY", 2, "C", "C", 4.190, 2.428, -0.481),
    ("GLY", 2, "O", "O", 3.418, 3.385, -0.481),
    ("GLY", 2, "H", "H", 1.760, 1.920, -0.470),
    ("GLY", 2, "HA2", "H", 4.020, 0.480, 0.400),
    ("GLY", 2, "HA3", "H", 4.020, 0.480, -1.360),
]
N_DIPEPTIDE_ATOMS = len(_DIPEPTIDE_ATOMS)
N_DIPEPTIDE_HEAVY_ATOMS = sum(1 for atom in _DIPEPTIDE_ATOMS if atom[3] != "H")


def _pdb_atom_line(serial: int, atom: tuple, b_factor: float) -> str:
    residue, residue_id, name, element, x, y, z = atom
    padded_name = f" {name:<3s}" if len(name) < 4 else name
    return (
        f"ATOM  {serial:5d} {padded_name} {residue:>3s} A{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b_factor:6.2f}          {element:>2s}"
    )


def write_pdb(path: Path, atoms: list[tuple], b_factor: float = 20.0) -> Path:
    """Write ``atoms`` as a minimal PDB file."""
    lines = [_pdb_atom_line(i + 1, atom, b_factor) for i, atom in enumerate(atoms)]
    path.write_text("\n".join([*lines, "END", ""]))
    return path


@pytest.fixture
def dipeptide_atoms() -> list[tuple]:
    """Atom records of the ALA-GLY test dipeptide."""
    return list(_DIPEPTIDE_ATOMS)


@pytest.fixture
def dipeptide_pdb(tmp_path: Path) -> Path:
    """Path to a PDB file with an ALA-GLY dipeptide including hydrogens."""
    return write_pdb(tmp_path / "dipeptide.pdb", _DIPEPTIDE_ATOMS)


def single_atom(
    element: str = "C",
    atomic_number: int = 6,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    b_factor: float = 20.0,
    occupancy: float = 1.0,
) -> AtomicStructure:
    """An ``AtomicStructure`` holding one atom."""
    return AtomicStructure(
        positions_zyx=torch.tensor([position]),
        atomic_numbers=torch.tensor([atomic_number]),
        elements=(element,),
        atom_names=(element,),
        b_factors=torch.tensor([b_factor]),
        occupancies=torch.tensor([occupancy]),
    )
