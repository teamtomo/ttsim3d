import mrcfile
import pytest
import torch
from pydantic import ValidationError
from torch_fourier_filter.mtf import get_mtf_reference_path

from ttsim3d import (
    DoseFilterConfig,
    PotentialConfig,
    Simulator,
    apply_fourier_filters,
    dose_filter_3d,
    simulate_potential,
)


@pytest.fixture
def simulator(dipeptide_pdb) -> Simulator:
    return Simulator(
        structure={"path": dipeptide_pdb},
        pixel_spacing=1.0,
        volume_shape=(24, 24, 24),
    )


def test_defaults(simulator):
    assert simulator.structure.center is True
    assert simulator.structure.remove_hydrogens is True
    assert simulator.potential == PotentialConfig()
    assert simulator.dose_filter == DoseFilterConfig()
    assert simulator.mtf is None


def test_old_schema_is_rejected(dipeptide_pdb):
    with pytest.raises(ValidationError):
        Simulator(
            pdb_filepath=dipeptide_pdb,
            pixel_spacing=1.0,
            volume_shape=(24, 24, 24),
            simulator_config={"voltage": 300.0},
        )


def test_mtf_validation(dipeptide_pdb, simulator):
    assert simulator.model_copy(update={"mtf": None}).mtf is None
    named = Simulator.model_validate(
        {**simulator.model_dump(), "mtf": "Falcon4EC_300kV"}
    )
    assert named.mtf == "falcon4ec_300kv"
    path = str(get_mtf_reference_path("k2_300kv"))
    assert Simulator.model_validate({**simulator.model_dump(), "mtf": path}).mtf == path
    with pytest.raises(ValidationError, match="mtf"):
        Simulator.model_validate({**simulator.model_dump(), "mtf": "not_a_camera"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 10.0, "end": 5.0},
        {"start": -1.0},
        {"critical_bfactor": 0.0},
        {"modify_signal": "None"},
    ],
)
def test_dose_filter_config_validation(kwargs):
    with pytest.raises(ValidationError):
        DoseFilterConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pixel_spacing": 0.0},
        {"volume_shape": (24, 24)},
        {"volume_shape": (24, -1, 24)},
        {"potential": {"upsampling": 0}},
        {"potential": {"sublattice_radius": 0.0}},
    ],
)
def test_simulator_validation(dipeptide_pdb, kwargs):
    settings = {
        "structure": {"path": dipeptide_pdb},
        "pixel_spacing": 1.0,
        "volume_shape": (24, 24, 24),
        **kwargs,
    }
    with pytest.raises(ValidationError):
        Simulator(**settings)


def test_yaml_round_trip(tmp_path, simulator):
    configured = simulator.model_copy(
        update={
            "mtf": "k3_standard_300kv_fl2",
            "dose_filter": DoseFilterConfig(end=50.0, modify_signal="rel_diff"),
            "potential": PotentialConfig(upsampling=2, sublattice_radius=4.0),
        }
    )
    path = tmp_path / "simulator.yaml"
    configured.to_yaml(path)
    assert Simulator.from_yaml(path) == configured


def test_run_matches_functional_pipeline(simulator):
    structure = simulator.load_structure()
    potential = simulate_potential(structure, 1.0, (24, 24, 24))
    expected = apply_fourier_filters(potential, dose_filter_3d((24, 24, 24), 1.0))
    assert torch.allclose(simulator.run(), expected)


def test_unfiltered_run_is_the_potential(simulator):
    unfiltered = simulator.model_copy(update={"dose_filter": None, "mtf": None})
    assert unfiltered.fourier_filters() == []
    assert torch.equal(unfiltered.run(), simulator.simulate_potential())


def test_b_factor_settings_are_applied(simulator):
    blurred = simulator.model_copy(deep=True)
    blurred.structure.b_factor_scale = 2.0
    blurred.structure.b_factor_offset = 50.0
    base = simulator.load_structure()
    adjusted = blurred.load_structure()
    assert torch.allclose(adjusted.b_factors, base.b_factors * 2.0 + 50.0)
    assert blurred.run().max() < simulator.run().max()


def test_pixel_spacing_can_change_between_runs(simulator):
    first = simulator.run()
    simulator.pixel_spacing = 1.5
    second = simulator.run()
    assert second.shape == first.shape
    assert not torch.allclose(first, second)


def test_export_to_mrc(tmp_path, simulator):
    path = tmp_path / "volume.mrc"
    simulator.export_to_mrc(path)
    with mrcfile.open(path) as mrc:
        assert mrc.data.shape == (24, 24, 24)
        assert float(mrc.voxel_size.x) == pytest.approx(1.0)
        assert mrc.data.dtype == "float32"
