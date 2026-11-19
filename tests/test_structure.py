import pytest
import torch

from tests.conftest import N_DIPEPTIDE_ATOMS, N_DIPEPTIDE_HEAVY_ATOMS, write_pdb
from ttsim3d import adjust_b_factors, load_structure


def test_hydrogens_removed_by_default(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb)
    assert structure.num_atoms == N_DIPEPTIDE_HEAVY_ATOMS
    assert "H" not in structure.elements
    assert not torch.any(structure.atomic_numbers == 1)


def test_keep_hydrogens(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb, remove_hydrogens=False)
    assert structure.num_atoms == N_DIPEPTIDE_ATOMS


def test_centering(dipeptide_pdb, dipeptide_atoms):
    centered = load_structure(dipeptide_pdb)
    assert torch.allclose(centered.positions_zyx.mean(dim=0), torch.zeros(3), atol=1e-5)

    raw = load_structure(dipeptide_pdb, center=False, remove_hydrogens=False)
    expected_zyx = torch.tensor([[z, y, x] for *_, x, y, z in dipeptide_atoms])
    assert torch.allclose(raw.positions_zyx, expected_zyx, atol=1e-3)


def test_fields_stay_aligned_after_hydrogen_removal(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb, annotate_bonding=True)
    n = structure.num_atoms
    assert n == N_DIPEPTIDE_HEAVY_ATOMS
    assert structure.atomic_numbers.shape == (n,)
    assert structure.b_factors.shape == (n,)
    assert structure.occupancies.shape == (n,)
    assert len(structure.elements) == len(structure.atom_names) == n
    assert structure.bonded_environments is not None
    assert structure.molecule_types is not None
    assert len(structure.bonded_environments) == len(structure.molecule_types) == n
    # Environment keys are annotated before hydrogens are dropped
    ca_index = structure.atom_names.index("CA")
    assert "H" in structure.bonded_environments[ca_index]
    assert set(structure.molecule_types) == {"protein"}


def test_only_hydrogens_raises(tmp_path, dipeptide_atoms):
    hydrogens = [atom for atom in dipeptide_atoms if atom[3] == "H"]
    path = write_pdb(tmp_path / "hydrogens.pdb", hydrogens)
    with pytest.raises(ValueError, match="No atoms left"):
        load_structure(path)


def test_adjust_b_factors(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb)
    assert adjust_b_factors(structure) is structure
    adjusted = adjust_b_factors(structure, scale=0.5, offset=10.0)
    assert torch.allclose(adjusted.b_factors, structure.b_factors * 0.5 + 10.0)
    assert torch.equal(adjusted.positions_zyx, structure.positions_zyx)
