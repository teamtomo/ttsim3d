"""Run tests for the Simulator and SimulatorConfig pydantic models."""

import pytest
from pydantic import ValidationError

from ttsim3d.models import DEFAULT_MTF_REFERENCES, SimulatorConfig


def good_simulator_config() -> dict:
    """Returns a dictionary of valid inputs for the SimulatorConfig model."""
    return {
        "voltage": 300,
        "apply_dose_weighting": True,
        "crit_exposure_bfactor": -1,
        "dose_filter_modify_signal": "None",
        "dose_start": 0.0,
        "dose_end": 30.0,
        "apply_dqe": True,
        "mtf_reference": "k2_300kv",
        "upsampling": -1,
        "store_volume": False,
    }


def test_default_mtf_references():
    """Ensure the default MTF references are valid."""
    good_dict = good_simulator_config()
    mtf_reference_strs = list(DEFAULT_MTF_REFERENCES.keys())

    # Ensure all default MTF references are valid
    for mtf_ref in mtf_reference_strs:
        good_dict["mtf_reference"] = mtf_ref
        _ = SimulatorConfig(**good_dict)


def test_simulator_config_validation():
    """Basic tests for the validation of SimulatorConfig model inputs."""
    good_dict = good_simulator_config()

    # First, check that a valid dictionary passes validation
    _ = SimulatorConfig(**good_dict)

    # Ensure voltage is positive
    bad_voltage = good_dict.copy()
    bad_voltage["voltage"] = -300
    with pytest.raises(ValidationError):
        SimulatorConfig(**bad_voltage)

    # Ensure the dose_filter_modify_signal is a valid choice
    bad_modify_signal = good_dict.copy()
    bad_modify_signal["dose_filter_modify_signal"] = "invalid_choice"
    with pytest.raises(ValidationError):
        SimulatorConfig(**bad_modify_signal)

    # Ensure invalid mtf paths raise an error
    bad_mtf = good_dict.copy()
    bad_mtf["mtf_reference"] = "/some/invalid/path/mtf.star"
    with pytest.raises(ValidationError):
        SimulatorConfig(**bad_mtf)


def test_simulator_validation():
    """Basic tests for the validation of Simulator model inputs."""
    # TODO: Implement test
    pass
