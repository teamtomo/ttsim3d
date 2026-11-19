import pytest
import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import get_mtf_reference_path, read_mtf
from torch_fourier_rescale import fourier_rescale_3d

from tests.conftest import single_atom
from ttsim3d import (
    apply_fourier_filters,
    dose_filter_3d,
    mtf_filter_3d,
    simulate_potential,
)

SHAPE = (16, 18, 20)


def test_dose_filter_matches_torch_fourier_filter():
    expected = cumulative_dose_filter_3d(
        volume_shape=SHAPE,
        pixel_size=1.2,
        start_exposure=5.0,
        end_exposure=40.0,
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
    )
    actual = dose_filter_3d(SHAPE, 1.2, dose_start=5.0, dose_end=40.0)
    assert actual.shape == (16, 18, 11)
    assert torch.allclose(actual, expected)

    with_bfactor = dose_filter_3d(SHAPE, 1.2, critical_bfactor=50.0)
    expected_bfactor = cumulative_dose_filter_3d(
        volume_shape=SHAPE, pixel_size=1.2, crit_exposure_bfactor=50.0
    )
    assert torch.allclose(with_bfactor, expected_bfactor)


def test_dose_filter_signal_modifications():
    base = dose_filter_3d(SHAPE, 1.0)
    assert torch.allclose(dose_filter_3d(SHAPE, 1.0, modify_signal="sqrt"), base.sqrt())
    assert torch.allclose(
        dose_filter_3d(SHAPE, 1.0, modify_signal="rel_diff"),
        1 - (1 - base) / (1 + base),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dose_start": -1.0}, "dose_start"),
        ({"dose_start": 10.0, "dose_end": 10.0}, "dose_end"),
        ({"critical_bfactor": 0.0}, "critical_bfactor"),
        ({"modify_signal": "cube"}, "modify_signal"),
    ],
)
def test_dose_filter_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        dose_filter_3d(SHAPE, 1.0, **kwargs)


def test_mtf_filter_inputs_are_equivalent():
    by_name = mtf_filter_3d(SHAPE, "K2_300kV")
    by_path = mtf_filter_3d(SHAPE, get_mtf_reference_path("k2_300kv"))
    by_tensors = mtf_filter_3d(SHAPE, read_mtf("k2_300kv"))
    assert by_name.shape == (16, 18, 11)
    assert torch.equal(by_name, by_path)
    assert torch.equal(by_name, by_tensors)
    assert float(by_name[0, 0, 0]) == pytest.approx(1.0, abs=1e-3)


def test_mtf_filter_rejects_mismatched_tensors():
    with pytest.raises(ValueError, match="same shape"):
        mtf_filter_3d(SHAPE, (torch.linspace(0, 0.5, 5), torch.ones(4)))


def test_apply_fourier_filters():
    volume = torch.randn(SHAPE)
    assert apply_fourier_filters(volume) is volume

    ones = torch.ones(16, 18, 11)
    assert torch.allclose(apply_fourier_filters(volume, ones), volume, atol=1e-5)

    dose = dose_filter_3d(SHAPE, 1.0)
    mtf = mtf_filter_3d(SHAPE, "k2_300kv")
    combined = apply_fourier_filters(volume, dose, mtf)
    sequential = apply_fourier_filters(apply_fourier_filters(volume, dose), mtf)
    assert torch.allclose(combined, sequential, atol=1e-5)


def test_dose_filter_commutes_with_fourier_crop():
    """Filtering after the crop equals filtering the upsampled volume first.

    ``Simulator`` relies on this to apply the dose filter at the output size.
    """
    upsampled = simulate_potential(single_atom(), 0.5, (64, 64, 64), upsampling=1)
    filter_then_crop, _ = fourier_rescale_3d(
        apply_fourier_filters(upsampled, dose_filter_3d((64, 64, 64), 0.5)),
        target_shape=(32, 32, 32),
    )
    cropped, _ = fourier_rescale_3d(upsampled, target_shape=(32, 32, 32))
    crop_then_filter = apply_fourier_filters(cropped, dose_filter_3d((32, 32, 32), 1.0))
    scale = float(crop_then_filter.abs().max())
    assert torch.allclose(filter_then_crop, crop_then_filter, atol=1e-4 * scale)
