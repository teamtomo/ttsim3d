"""Test for functions defined in src/ttsim3d/simulate3d.py"""

import pytest
import torch

from ttsim3d.simulate3d import _validate_dose_filter_inputs, _validate_dqe_filter_inputs


##################################
### Tests for helper functions ###
##################################
def test_calculate_lead_term():
    pass


def test_validate_dose_filter_inputs():
    """Basic tests for validate_dose_filter_inputs method."""
    # Ensure no errors are raised when inputs are valid
    permitted_modify_signals = ["None", "sqrt", "rel_diff"]
    for mf in permitted_modify_signals:
        _validate_dose_filter_inputs(
            dose_filter_modify_signal=mf,
            dose_filter_critical_bfactor=-1,  # Grant-Grigorieff 2015
        )
        _validate_dose_filter_inputs(
            dose_filter_modify_signal=mf,
            dose_filter_critical_bfactor=30.0,
        )

    # Ensure errors are raised when inputs are invalid
    with pytest.raises(ValueError):
        _validate_dose_filter_inputs(
            dose_filter_modify_signal="invalid",  # bad parameter
            dose_filter_critical_bfactor=-1,
        )

    with pytest.raises(ValueError):
        _validate_dose_filter_inputs(
            dose_filter_modify_signal="None",
            dose_filter_critical_bfactor=-2.5,  # bad parameter
        )


def test_validate_dqe_filter_inputs():
    """Basic tests for validate_dqe_filter_inputs method."""
    _validate_dqe_filter_inputs(
        apply_dqe=False,
        mtf_frequencies=None,
        mtf_amplitudes=None,
    )
    _validate_dqe_filter_inputs(
        apply_dqe=True,
        mtf_frequencies=torch.tensor([0.0, 0.25, 0.5]),
        mtf_amplitudes=torch.tensor([1.0, 0.5, 0.0]),
    )

    # Errors raised when mtf tensors are none
    with pytest.raises(ValueError):
        _validate_dqe_filter_inputs(
            apply_dqe=True,
            mtf_frequencies=None,
            mtf_amplitudes=torch.tensor([1.0, 0.5, 0.0]),
        )
    with pytest.raises(ValueError):
        _validate_dqe_filter_inputs(
            apply_dqe=True,
            mtf_frequencies=torch.tensor([0.0, 0.25, 0.5]),
            mtf_amplitudes=None,
        )

    # Errors raised when mtf tensors are not the same size
    with pytest.raises(ValueError):
        _validate_dqe_filter_inputs(
            apply_dqe=True,
            mtf_frequencies=torch.tensor([0.0, 0.25, 0.5]),
            mtf_amplitudes=torch.tensor([1.0, 0.5]),
        )


def test_setup_sim3d_upsampling():
    pass


def test_setup_upsampling_coords():
    pass


###############################################
### Tests for the core simulation functions ###
###############################################
def test_simulate_atomwise_scattering_potentials():
    pass


def test_place_voxel_neighborhoods_in_volume():
    pass


def test_calculate_simulation_dose_filter_3d():
    pass


def test_apply_simulation_filters():
    pass


def test_simulate3d():
    pass
