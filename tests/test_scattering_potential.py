import torch

from ttsim3d.scattering_potential import (
    get_a_param,
    get_b_param,
    get_total_b_param,
)


def test_get_scattering_parameters():
    """TODO: Construct this test"""
    pass


def test_get_a_param():
    """Test the get_a_param function."""
    atom_ids = ["H", "N", "O", "C"]

    # NOTE: These are just copied from the elastic_scattering_factors.json file
    expected = torch.tensor(
        [
            [0.0349, 0.1201, 0.1970, 0.0573, 0.1195],
            [0.1022, 0.3219, 0.7982, 0.8197, 0.1715],
            [0.0974, 0.2921, 0.6910, 0.6990, 0.2039],
            [0.0893, 0.2563, 0.7570, 1.0487, 0.3575],
        ]
    )
    result = get_a_param(atom_ids)

    assert isinstance(result, torch.Tensor), f"Expected a tensor, got {type(result)}"
    assert torch.allclose(
        result, expected
    ), "Mismatch between expected a parameters and result"


def test_get_b_param():
    """Test the get_b_param function."""
    atom_ids = ["H", "N", "O", "C"]

    # NOTE: These are just copied from the elastic_scattering_factors.json file
    expected = torch.tensor(
        [
            [0.5347, 3.5867, 12.3471, 18.9525, 38.6269],
            [0.2451, 1.7481, 6.1925, 17.3894, 48.1431],
            [0.2067, 1.3815, 4.6943, 12.7105, 32.4726],
            [0.2465, 1.7100, 6.4094, 18.6113, 50.2523],
        ]
    )

    result = get_b_param(atom_ids)

    assert isinstance(result, torch.Tensor), f"Expected a tensor, got {type(result)}"

    assert torch.allclose(
        result, expected
    ), "Mismatch between expected b parameters and result"


def test_get_total_b_param():
    """Test for combined b parameter and B-factor."""
    atom_ids = ["H", "N", "O", "C"]
    atom_b_factors = torch.tensor([40.0, 30.0, 20.0, 10.0])

    # expected is 2 * PI / sqrt(B_n + b)
    expected = torch.tensor(
        [
            [0.986885, 0.951706, 0.868428, 0.818331, 0.708589],
            [1.142490, 1.115119, 1.044409, 0.912724, 0.710779],
            [1.397759, 1.358816, 1.264391, 1.098592, 0.867388],
            [1.962873, 1.836122, 1.551078, 1.174657, 0.809456],
        ]
    )

    result = get_total_b_param(atom_ids, atom_b_factors)

    assert isinstance(result, torch.Tensor), f"Expected a tensor, got {type(result)}"

    assert torch.allclose(
        result, expected
    ), "Mismatch between expected total b parameters and result"
