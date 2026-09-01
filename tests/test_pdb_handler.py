import pathlib
import tempfile

import pytest
import torch

from ttsim3d.pdb_handler import load_model, remove_hydrogens


class TestLoadModel:
    """Tests for load_model function."""

    @pytest.fixture
    def sample_pdb_file(self):
        """Create a temporary PDB file for testing."""
        pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 15.00           C
ATOM      3  C   ALA A   1       2.009   1.390   0.000  1.00 12.00           C
ATOM      4  H   ALA A   1      -0.500  -0.500  -0.500  1.00 20.00           H
END
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        yield temp_path
        pathlib.Path(temp_path).unlink()

    def test_load_model_returns_correct_types(self, sample_pdb_file):
        """Test that load_model returns correct data types."""
        atom_zyx, atom_id, atom_b_factor = load_model(sample_pdb_file)

        assert isinstance(atom_zyx, torch.Tensor)
        assert isinstance(atom_id, list)
        assert isinstance(atom_b_factor, torch.Tensor)

    def test_load_model_correct_shapes(self, sample_pdb_file):
        """Test that load_model returns correct tensor shapes."""
        atom_zyx, atom_id, atom_b_factor = load_model(sample_pdb_file)

        n_atoms = len(atom_id)
        assert atom_zyx.shape == (n_atoms, 3)
        assert atom_b_factor.shape == (n_atoms,)

    def test_load_model_centers_atoms(self, sample_pdb_file):
        """Test that atoms are centered at origin when center_atoms=True."""
        atom_zyx, _, _ = load_model(sample_pdb_file, center_atoms=True)

        # Mean should be close to zero
        mean = torch.mean(atom_zyx, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)

    def test_load_model_no_centering(self, sample_pdb_file):
        """Test that atoms are not centered when center_atoms=False."""
        atom_zyx_centered, _, _ = load_model(sample_pdb_file, center_atoms=True)
        atom_zyx_uncentered, _, _ = load_model(sample_pdb_file, center_atoms=False)

        # Uncentered should not have mean near zero
        mean_uncentered = torch.mean(atom_zyx_uncentered, dim=0)
        assert not torch.allclose(mean_uncentered, torch.zeros(3), atol=1e-6)

        # But they should be offset versions of each other
        center_offset = torch.mean(atom_zyx_uncentered, dim=0)
        assert torch.allclose(
            atom_zyx_centered, atom_zyx_uncentered - center_offset, atol=1e-5
        )

    def test_load_model_atom_ids(self, sample_pdb_file):
        """Test that atom IDs are extracted correctly."""
        _, atom_id, _ = load_model(sample_pdb_file)

        # Should have 4 atoms including hydrogen
        assert len(atom_id) == 4
        # Check that elements are uppercase strings
        assert all(isinstance(aid, str) for aid in atom_id)
        assert "H" in atom_id

    def test_load_model_b_factors(self, sample_pdb_file):
        """Test that B factors are loaded correctly."""
        _, _, atom_b_factor = load_model(sample_pdb_file)

        # B factors should be float tensors
        assert atom_b_factor.dtype == torch.float32
        # Should have reasonable values (10, 15, 12, 20 from the PDB)
        assert torch.allclose(atom_b_factor[0], torch.tensor(10.0), atol=1e-5)


class TestRemoveHydrogens:
    """Tests for remove_hydrogens function."""

    def test_remove_hydrogens_filters_correctly(self):
        """Test that hydrogen atoms are removed."""
        atoms_zyx = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        atoms_id = ["N", "C", "H", "O"]
        atoms_b_factor = torch.tensor([10.0, 15.0, 20.0, 12.0])

        zyx_filtered, id_filtered, b_filtered = remove_hydrogens(
            atoms_zyx, atoms_id, atoms_b_factor
        )

        assert len(id_filtered) == 3
        assert "H" not in id_filtered
        assert id_filtered == ["N", "C", "O"]

    def test_remove_hydrogens_maintains_order(self):
        """Test that removal maintains atom order."""
        atoms_zyx = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        atoms_id = ["N", "H", "C"]
        atoms_b_factor = torch.tensor([10.0, 20.0, 15.0])

        zyx_filtered, id_filtered, b_filtered = remove_hydrogens(
            atoms_zyx, atoms_id, atoms_b_factor
        )

        assert id_filtered == ["N", "C"]
        assert torch.allclose(zyx_filtered[0], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(zyx_filtered[1], torch.tensor([2.0, 2.0, 2.0]))

    def test_remove_hydrogens_correct_tensors(self):
        """Test that tensor shapes are correct after filtering."""
        atoms_zyx = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        atoms_id = ["N", "H", "C"]
        atoms_b_factor = torch.tensor([10.0, 20.0, 15.0])

        zyx_filtered, id_filtered, b_filtered = remove_hydrogens(
            atoms_zyx, atoms_id, atoms_b_factor
        )

        assert zyx_filtered.shape == (2, 3)
        assert b_filtered.shape == (2,)

    def test_remove_hydrogens_no_hydrogens(self):
        """Test behavior when there are no hydrogens."""
        atoms_zyx = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        atoms_id = ["N", "C"]
        atoms_b_factor = torch.tensor([10.0, 15.0])

        zyx_filtered, id_filtered, b_filtered = remove_hydrogens(
            atoms_zyx, atoms_id, atoms_b_factor
        )

        assert len(id_filtered) == 2
        assert torch.equal(zyx_filtered, atoms_zyx)
        assert torch.equal(b_filtered, atoms_b_factor)

    def test_remove_hydrogens_only_hydrogens(self):
        """Test behavior when all atoms are hydrogens."""
        atoms_zyx = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        atoms_id = ["H", "H"]
        atoms_b_factor = torch.tensor([10.0, 15.0])

        zyx_filtered, id_filtered, b_filtered = remove_hydrogens(
            atoms_zyx, atoms_id, atoms_b_factor
        )

        assert len(id_filtered) == 0
        assert zyx_filtered.shape == (0, 3)
        assert b_filtered.shape == (0,)
