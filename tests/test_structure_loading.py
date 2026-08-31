"""Tests for monorepo-backed structure loading."""

from pathlib import Path

import pandas as pd
import torch
from torch_structure_manipulation import annotate_bonding_environments

from ttsim3d.models import Simulator, SimulatorConfig
from ttsim3d.structure_loading import StructureLoadOptions, df_params_to_tensors, load_structure


def _mini_atoms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", 1, "ALA", "CA", "C", 0.0, 0.0, 0.0, 10.0),
            ("A", 1, "ALA", "N", "N", 1.0, 0.0, 0.0, 12.0),
            ("B", 1, "U", "C2", "C", 2.0, 0.0, 0.0, 11.0),
        ],
        columns=[
            "chain",
            "residue_id",
            "residue",
            "atom",
            "element",
            "x",
            "y",
            "z",
            "b_isotropic",
        ],
    )


def _write_mini_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  N   ALA A   1       1.000   0.000   0.000  1.00 12.00           N",
                "ATOM      3  C2  U   B   1       2.000   0.000   0.000  1.00 11.00           C",
                "END",
            ]
        )
    )


def test_load_structure_annotates_bonding(tmp_path: Path):
    pdb_path = tmp_path / "mini.pdb"
    _write_mini_pdb(pdb_path)

    df = load_structure(
        pdb_path,
        StructureLoadOptions(
            center_atoms=True,
            include_hydrogens=False,
            load_bonded_environment=True,
        ),
    )

    assert "bonded_environment" in df.columns
    assert "molecule_type" in df.columns
    assert set(df["molecule_type"]) <= {"protein", "rna", "other"}


def test_df_params_to_tensors_matches_simulator_fields():
    df = annotate_bonding_environments(_mini_atoms(), include_hydrogens=False)
    positions, atom_ids, b_factors, bonded_ids, molecule_types = df_params_to_tensors(
        df
    )

    assert positions.shape == (3, 3)
    assert atom_ids == ["C", "N", "C"]
    assert torch.allclose(b_factors, torch.tensor([10.0, 12.0, 11.0]))
    assert bonded_ids is not None and len(bonded_ids) == 3
    assert molecule_types is not None and len(molecule_types) == 3


def test_simulator_loads_bonded_structure_from_file(tmp_path: Path):
    pdb_path = tmp_path / "mini.pdb"
    _write_mini_pdb(pdb_path)

    config = SimulatorConfig(
        apply_dose_weighting=False,
        apply_dqe=False,
        store_volume=False,
        use_bonded_scattering_factors=True,
    )
    sim = Simulator(
        pdb_filepath=pdb_path,
        pixel_spacing=2.0,
        volume_shape=(16, 16, 16),
        simulator_config=config,
        remove_hydrogens=True,
    )

    assert sim.atom_positions_zyx.shape == (3, 3)
    assert sim.atom_bonded_ids is not None
    assert sim.molecule_type is not None
