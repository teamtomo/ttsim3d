import math

import pytest
import torch
from torch_calculate_electrostatic_potential import (
    PENG_SCATTERING_TO_POTENTIAL,
    get_peng_scattering_parameters,
)

from tests.conftest import single_atom
from ttsim3d import (
    auto_sublattice_radius,
    auto_upsampling,
    load_structure,
    simulate_potential,
)


def _peak(volume: torch.Tensor) -> list[int]:
    return [int(i) for i in torch.unravel_index(volume.argmax(), volume.shape)]


def _carbon_integral() -> float:
    """Expected sum(V) * voxel_volume for one carbon atom, in V * A^3."""
    params_a, _ = get_peng_scattering_parameters(torch.tensor([6]))
    return PENG_SCATTERING_TO_POTENTIAL * float(params_a.sum())


@pytest.mark.parametrize("shape", [(32, 32, 32), (33, 33, 33), (31, 32, 33)])
@pytest.mark.parametrize("upsampling", [1, 2, 4])
def test_origin_is_at_voxel_n_over_2(shape, upsampling):
    volume = simulate_potential(single_atom(), 1.0, shape, upsampling=upsampling)
    assert volume.shape == shape
    assert _peak(volume) == [n // 2 for n in shape]


@pytest.mark.parametrize("upsampling", [1, 2, 4])
def test_potential_is_in_volts(upsampling):
    pixel_spacing = 0.8
    volume = simulate_potential(
        single_atom(), pixel_spacing, (40, 40, 40), upsampling=upsampling
    )
    integral = float(volume.sum()) * pixel_spacing**3
    assert integral == pytest.approx(_carbon_integral(), rel=1e-3)


def test_occupancy_weights_atoms():
    full = simulate_potential(single_atom(), 1.0, (24, 24, 24), upsampling=1)
    half = simulate_potential(
        single_atom(occupancy=0.5), 1.0, (24, 24, 24), upsampling=1
    )
    assert torch.allclose(half, 0.5 * full)


@pytest.mark.parametrize("b_factor", [0.0, 30.0, 150.0])
def test_auto_sublattice_radius_captures_potential(b_factor):
    atom = single_atom(b_factor=b_factor)
    auto = simulate_potential(atom, 0.5, (64, 64, 64), upsampling=1)
    wide = simulate_potential(
        atom, 0.5, (64, 64, 64), upsampling=1, sublattice_radius=15.0
    )
    assert float(auto.sum()) / float(wide.sum()) > 1 - 1e-3


def test_auto_sublattice_radius_formula():
    params_b = torch.tensor([[0.5, 2.0, 10.0, 30.0, 50.0]])
    b_factors = torch.tensor([10.0, 30.0])
    expected = math.sqrt(80.0 * math.log(1e3)) / (2 * math.pi)
    assert auto_sublattice_radius(params_b, b_factors) == pytest.approx(expected)
    with pytest.raises(ValueError, match="rel_tol"):
        auto_sublattice_radius(params_b, b_factors, rel_tol=1.0)


@pytest.mark.parametrize(
    ("pixel_spacing", "edge", "expected"),
    [
        (2.0, 64, 4),
        (2.0, 400, 2),  # 4x would exceed the size limit
        (1.6, 1000, 1),
        (1.0, 400, 2),
        (1.0, 800, 1),
        (0.75, 64, 1),
        (0.5, 64, 1),
    ],
)
def test_auto_upsampling(pixel_spacing, edge, expected):
    assert auto_upsampling(pixel_spacing, (edge, edge, edge)) == expected


def test_batched_positions():
    atom = single_atom()
    shifted = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]])
    batched = atom.with_positions(shifted)
    volume = simulate_potential(batched, 1.0, (16, 16, 16), upsampling=1)
    assert volume.shape == (2, 16, 16, 16)
    assert _peak(volume[0]) == [8, 8, 8]
    assert _peak(volume[1]) == [8, 8, 10]


def test_bonded_factors_require_annotation(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb)
    with pytest.raises(ValueError, match="annotate_bonding=True"):
        simulate_potential(
            structure, 1.0, (16, 16, 16), scattering_factors="peng_bonded"
        )


def test_bonded_factors_change_the_potential(dipeptide_pdb):
    structure = load_structure(dipeptide_pdb, annotate_bonding=True)
    elemental = simulate_potential(structure, 1.0, (24, 24, 24))
    # The chain termini have no tabulated bonded environment
    with pytest.warns(UserWarning, match="unsupported bonded scattering"):
        bonded = simulate_potential(
            structure, 1.0, (24, 24, 24), scattering_factors="peng_bonded"
        )
    assert bonded.shape == elemental.shape
    assert not torch.allclose(bonded, elemental)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pixel_spacing": 0.0}, "pixel_spacing"),
        ({"volume_shape": (16, 16)}, "volume_shape"),
        ({"volume_shape": (16, 0, 16)}, "volume_shape"),
        ({"upsampling": 0}, "upsampling"),
        ({"sublattice_radius": -1.0}, "sublattice_radius"),
    ],
)
def test_invalid_arguments(kwargs, match):
    arguments = {"pixel_spacing": 1.0, "volume_shape": (16, 16, 16), **kwargs}
    with pytest.raises(ValueError, match=match):
        simulate_potential(single_atom(), **arguments)
