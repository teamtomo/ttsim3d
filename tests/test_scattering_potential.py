import gemmi
import torch
from torch_calculate_electrostatic_potential import get_peng_scattering_parameters
from torch_structure_manipulation import annotate_bonding_environments

from ttsim3d.scattering_potential import (
    get_a_param,
    get_b_param,
    get_total_b_param,
    resolve_atom_scattering_parameters,
)


def _elemental_expected(atom_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor(
        [gemmi.Element(atom_id).atomic_number for atom_id in atom_ids],
        dtype=torch.int64,
    )
    return get_peng_scattering_parameters(atomic_numbers)


def test_get_a_param():
    """Elemental parameters match the bundled Peng table."""
    atom_ids = ["H", "N", "O", "C"]
    expected_a, _ = _elemental_expected(atom_ids)
    result = get_a_param(atom_ids)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected_a)


def test_get_b_param():
    """Elemental width parameters match the bundled Peng table."""
    atom_ids = ["H", "N", "O", "C"]
    _, expected_b = _elemental_expected(atom_ids)
    result = get_b_param(atom_ids)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected_b)


def test_get_total_b_param():
    """Combined B-parameter kernel matches the historical formula."""
    atom_ids = ["H", "N", "O", "C"]
    atom_b_factors = torch.tensor([40.0, 30.0, 20.0, 10.0])
    expected = torch.tensor(
        [
            [0.986885, 0.951706, 0.868428, 0.818331, 0.708589],
            [1.142490, 1.115119, 1.044409, 0.912724, 0.710779],
            [1.397759, 1.358816, 1.264391, 1.098592, 0.867388],
            [1.962873, 1.836122, 1.551078, 1.174657, 0.809456],
        ]
    )

    result = get_total_b_param(atom_ids, atom_b_factors)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected)


def test_bonded_parameters_use_monorepo_tables():
    import pandas as pd

    atoms = pd.DataFrame(
        [
            ("A", 1, "ALA", "C", "C", 0.0, 0.0, 0.0),
            ("A", 1, "ALA", "O", "O", 1.0, 0.0, 0.0),
            ("A", 1, "ALA", "CA", "C", -1.0, 0.0, 0.0),
        ],
        columns=["chain", "residue_id", "residue", "atom", "element", "x", "y", "z"],
    )
    annotated = annotate_bonding_environments(atoms, include_hydrogens=False)
    atom_ids = annotated["element"].str.upper().tolist()
    bonded_ids = annotated["bonded_environments"].tolist()
    molecule_types = annotated["molecule_type"].tolist()

    params_a, params_b = resolve_atom_scattering_parameters(
        atom_ids,
        atom_bonded_ids=bonded_ids,
        molecule_type=molecule_types,
    )

    assert params_a.shape == (3, 5)
    assert params_b.shape == (3, 5)
    elemental_a, _ = _elemental_expected(atom_ids)
    assert not torch.allclose(params_a, elemental_a)
