"""Tests for the 'matched' dose filter and the ttsim3d-refilter program."""

import json

import mrcfile
import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pydantic import ValidationError
from src.programs.run_refilter import refilter_cli
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d

from ttsim3d.envelope import apply_template_envelope, refilter_mrc
from ttsim3d.models import Simulator, SimulatorConfig
from ttsim3d.simulate3d import (
    _validate_dose_filter_inputs,
    calculate_simulation_dose_filter_3d,
    simulate3d,
)

# TODO: drop the skip once torch-fourier-filter is pinned to the release that
# contains teamtomo/teamtomo#203.
fw = pytest.importorskip("torch_fourier_filter.frame_weights")

SHAPE = (24, 24, 24)
PIXEL = 0.5


@pytest.fixture
def table_json(tmp_path):
    """A frame weight table over 0-40 e/A^2 that differs from the default curve."""
    start = torch.arange(40, dtype=torch.float64)
    table = fw.FrameWeightTable(
        exposure_start=start,
        exposure_end=start + 1,
        bfactor=-8.0 * start,
        log_scale=-0.01 * start,
        source="estimated",
    )
    path = tmp_path / "frame_weights.json"
    table.to_json(path)
    return path


@pytest.fixture
def envelope_json(tmp_path):
    """A template envelope with a non-flat correction."""
    start, end = fw.frame_exposures(25, 0.0, 1.0)
    envelope = fw.TemplateEnvelope(
        frame_weights=fw.default_frame_bfactors(start, end),
        correction=fw.EnvelopeCorrection.uniform_knots(
            log_values=[0.0, 0.1, -0.1, 0.2, -0.3]
        ),
    )
    path = tmp_path / "template_envelope.json"
    envelope.to_json(path)
    return path


def _matched(dose_start, dose_end, **kwargs):
    return calculate_simulation_dose_filter_3d(
        shape=kwargs.pop("shape", SHAPE),
        pixel_spacing=PIXEL,
        dose_start=dose_start,
        dose_end=dose_end,
        critical_bfactor=-1,
        modify_signal="matched",
        **kwargs,
    )


#########################
### The matched filter ###
#########################


@pytest.mark.parametrize(
    ("dose_start", "dose_end"), [(0.0, 30.0), (2.0, 37.5), (0.0, 0.4)]
)
@pytest.mark.parametrize("frame_weights", [None, "default"])
def test_matched_default_equals_template_envelope_default(
    dose_start, dose_end, frame_weights
):
    dose_filter = _matched(dose_start, dose_end, frame_weights=frame_weights)
    expected = fw.TemplateEnvelope.default(dose_start, dose_end).grid(
        SHAPE, PIXEL, rfft=True
    )
    assert dose_filter.shape == (24, 24, 13)
    torch.testing.assert_close(dose_filter, expected, rtol=0, atol=0)
    assert dose_filter[0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(("rfft", "fftshift"), [(False, True), (True, True)])
def test_matched_grid_layout(rfft, fftshift):
    shape = (20, 22, 24)
    dose_filter = _matched(0.0, 30.0, shape=shape, rfft=rfft, fftshift=fftshift)
    expected = fw.TemplateEnvelope.default(0.0, 30.0).grid(
        shape, PIXEL, rfft=rfft, fftshift=fftshift
    )
    torch.testing.assert_close(dose_filter, expected, rtol=0, atol=0)


def test_matched_frame_weights_file(table_json):
    dose_filter = _matched(0.0, 30.0, frame_weights=str(table_json))

    # 30 frames of 1 e/A^2, written out independently of ttsim3d's helper
    start = torch.arange(30, dtype=torch.float64)
    table = fw.FrameWeightTable.from_json(table_json).resample(start, start + 1)
    expected = fw.TemplateEnvelope(frame_weights=table).grid(SHAPE, PIXEL)
    torch.testing.assert_close(dose_filter, expected, rtol=0, atol=0)

    # ... and the file is really used
    default = _matched(0.0, 30.0)
    assert not torch.allclose(dose_filter, default, atol=1e-3)


def test_matched_frame_weights_file_outside_table_warns(table_json):
    with pytest.warns(UserWarning, match="runs past the frame weight table"):
        _matched(0.0, 45.0, frame_weights=str(table_json))


def test_matched_template_envelope_file(envelope_json, table_json):
    expected = fw.TemplateEnvelope.from_json(envelope_json).grid(SHAPE, PIXEL)

    # The envelope holds its own frames: the dose range is not used ...
    torch.testing.assert_close(
        _matched(0.0, 30.0, template_envelope=str(envelope_json)),
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _matched(5.0, 5.0, template_envelope=envelope_json), expected, rtol=0, atol=0
    )
    # ... and it takes precedence over frame weights.
    torch.testing.assert_close(
        _matched(0.0, 30.0, frame_weights=table_json, template_envelope=envelope_json),
        expected,
        rtol=0,
        atol=0,
    )


def test_matched_needs_a_dose_range():
    with pytest.raises(ValueError, match="dose_end > dose_start"):
        _matched(10.0, 10.0)


@pytest.mark.parametrize("modify_signal", ["None", "sqrt", "rel_diff"])
@pytest.mark.parametrize("critical_bfactor", [-1, 30.0])
def test_grant_grigorieff_modes_unchanged(modify_signal, critical_bfactor):
    dose_filter = calculate_simulation_dose_filter_3d(
        shape=SHAPE,
        pixel_spacing=PIXEL,
        dose_start=0.0,
        dose_end=30.0,
        critical_bfactor=critical_bfactor,
        modify_signal=modify_signal,
    )
    gg = cumulative_dose_filter_3d(
        volume_shape=SHAPE,
        pixel_size=PIXEL,
        start_exposure=0.0,
        end_exposure=30.0,
        crit_exposure_bfactor=critical_bfactor,
        rfft=True,
        fftshift=False,
    )
    expected = {
        "None": gg,
        "sqrt": torch.sqrt(gg),
        "rel_diff": 1 - (1 - gg) / (1 + gg),
    }[modify_signal]
    torch.testing.assert_close(dose_filter, expected, rtol=0, atol=0)


def test_validate_dose_filter_inputs_matched():
    _validate_dose_filter_inputs("matched", -1)
    _validate_dose_filter_inputs("matched", -1, "default", None)
    with pytest.raises(ValueError, match="only used by the 'matched'"):
        _validate_dose_filter_inputs("None", -1, "default", None)
    with pytest.raises(ValueError, match="only used by the 'matched'"):
        _validate_dose_filter_inputs("sqrt", -1, None, "envelope.json")


def _toy_atoms():
    torch.manual_seed(0)
    positions = (torch.rand(12, 3) - 0.5) * 8.0
    ids = ["C", "N", "O", "S"] * 3
    b_factors = torch.full((12,), 5.0)
    return positions, ids, b_factors


@pytest.mark.parametrize("voltage", [200.0, 300.0])
def test_simulate3d_matched_equals_refilter(voltage):
    """'matched' in the simulation equals the envelope applied afterwards.

    With upsampling 1 the dose filter is applied on the final grid, so a matched
    simulation is an unweighted one times the envelope. The same envelope at both
    voltages: exposures are physical, with no 200 kV factor.
    """
    positions, ids, b_factors = _toy_atoms()
    kwargs = {
        "atom_positions_zyx": positions,
        "atom_ids": ids,
        "atom_b_factors": b_factors,
        "beam_energy_kev": voltage,
        "sim_pixel_spacing": 1.0,
        "sim_volume_shape": (20, 20, 20),
        "requested_upsampling": 1,
        "dose_start": 0.0,
        "dose_end": 25.0,
        "apply_dqe": False,
        "device": "cpu",
    }
    plain = simulate3d(apply_dose_weighting=False, **kwargs)
    matched = simulate3d(
        apply_dose_weighting=True, dose_filter_modify_signal="matched", **kwargs
    )
    envelope = fw.TemplateEnvelope.default(0.0, 25.0)
    expected = torch.fft.irfftn(
        torch.fft.rfftn(plain) * envelope.grid((20, 20, 20), 1.0), s=(20, 20, 20)
    )
    torch.testing.assert_close(matched, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(matched, plain, atol=1e-4)


def _write_toy_pdb(path):
    lines = []
    positions, ids, _ = _toy_atoms()
    for i, ((z, y, x), element) in enumerate(zip(positions.tolist(), ids), start=1):
        lines.append(
            f"ATOM  {i:5d}  {element:<3} ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2}"
        )
    path.write_text("\n".join([*lines, "END", ""]))
    return path


def test_simulator_passes_matched_inputs(tmp_path, envelope_json):
    pdb = _write_toy_pdb(tmp_path / "toy.pdb")
    common = {"apply_dqe": False, "upsampling": 1, "store_volume": False}
    sim_kwargs = {"pdb_filepath": pdb, "pixel_spacing": 1.0, "volume_shape": (20,) * 3}

    plain = Simulator(
        simulator_config=SimulatorConfig(apply_dose_weighting=False, **common),
        **sim_kwargs,
    ).run()
    matched = Simulator(
        simulator_config=SimulatorConfig(
            dose_filter_modify_signal="matched",
            template_envelope=str(envelope_json),
            **common,
        ),
        **sim_kwargs,
    ).run()

    assert plain.abs().max() > 1e-3  # the toy atoms were simulated

    envelope = fw.TemplateEnvelope.from_json(envelope_json)
    torch.testing.assert_close(
        matched, apply_template_envelope(plain, envelope, 1.0), rtol=1e-5, atol=1e-6
    )


##################
### Refiltering ###
##################


def test_apply_template_envelope_matches_direct_multiply():
    torch.manual_seed(1)
    shape = (20, 22, 24)
    volume = torch.randn(shape)
    envelope = fw.TemplateEnvelope.default(0.0, 40.0)

    filtered = apply_template_envelope(volume, envelope, 1.2)

    direct = torch.fft.irfftn(
        torch.fft.rfftn(volume) * envelope.grid(shape, 1.2, rfft=True), s=shape
    )
    torch.testing.assert_close(filtered, direct, rtol=0, atol=0)

    # An independent grid, as in a hand-written refilter: |k| in 1/A from
    # fftfreq / rfftfreq, with the envelope evaluated at every voxel
    k = (
        torch.fft.fftfreq(shape[0])[:, None, None] ** 2
        + torch.fft.fftfreq(shape[1])[None, :, None] ** 2
        + torch.fft.rfftfreq(shape[2])[None, None, :] ** 2
    ).sqrt() / 1.2
    by_voxel = torch.fft.irfftn(torch.fft.rfftn(volume) * envelope(k), s=shape)
    torch.testing.assert_close(filtered, by_voxel, rtol=1e-4, atol=1e-4)


def _write_toy_mrc(path, voxel_size=1.5, shape=(16, 18, 20)):
    rng = np.random.default_rng(2)
    data = rng.standard_normal(shape).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
    return torch.from_numpy(data)


def _direct(volume, envelope, pixel):
    shape = tuple(volume.shape)
    return torch.fft.irfftn(
        torch.fft.rfftn(volume) * envelope.grid(shape, pixel), s=shape
    )


def _read_mrc(path):
    with mrcfile.open(path) as mrc:
        return torch.from_numpy(mrc.data.copy()), float(mrc.voxel_size.x)


def test_refilter_cli_frame_weights(tmp_path):
    volume = _write_toy_mrc(tmp_path / "in.mrc")
    out = tmp_path / "out.mrc"
    result = CliRunner().invoke(
        refilter_cli,
        [
            "--input-mrc", str(tmp_path / "in.mrc"),
            "--output-mrc", str(out),
            "--frame-weights", "default",
            "--dose-start", "0",
            "--dose-end", "20",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    filtered, voxel_size = _read_mrc(out)
    assert voxel_size == pytest.approx(1.5)
    expected = _direct(volume, fw.TemplateEnvelope.default(0.0, 20.0), 1.5)
    torch.testing.assert_close(filtered, expected, rtol=1e-5, atol=1e-6)


def test_refilter_cli_template_envelope(tmp_path, envelope_json):
    volume = _write_toy_mrc(tmp_path / "in.mrc", voxel_size=0.95)
    out = tmp_path / "out.mrc"
    result = CliRunner().invoke(
        refilter_cli,
        [
            "--input-mrc", str(tmp_path / "in.mrc"),
            "--output-mrc", str(out),
            "--template-envelope", str(envelope_json),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    filtered, voxel_size = _read_mrc(out)
    assert voxel_size == pytest.approx(0.95)
    expected = _direct(volume, fw.TemplateEnvelope.from_json(envelope_json), 0.95)
    torch.testing.assert_close(filtered, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "extra",
    [
        ["--frame-weights", "default"],  # both sources
        ["--dose-end", "40"],  # a dose range the envelope does not use
        ["--frame-weights", "defualt"],
    ],
)
def test_refilter_cli_rejects_bad_combinations(tmp_path, envelope_json, extra):
    _write_toy_mrc(tmp_path / "in.mrc")
    args = ["--input-mrc", str(tmp_path / "in.mrc"), "--output-mrc", "x.mrc"]
    if extra[1] != "defualt":
        args += ["--template-envelope", str(envelope_json)]
    result = CliRunner().invoke(refilter_cli, args + extra)
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "x.mrc").exists()


def test_refilter_mrc_voxel_size(tmp_path):
    volume = _write_toy_mrc(tmp_path / "in.mrc", voxel_size=None)
    envelope = fw.TemplateEnvelope.default(0.0, 30.0)
    with pytest.raises(ValueError, match="no voxel size"):
        refilter_mrc(tmp_path / "in.mrc", tmp_path / "out.mrc", envelope)

    filtered = refilter_mrc(
        tmp_path / "in.mrc", tmp_path / "out.mrc", envelope, pixel_spacing=2.0
    )
    torch.testing.assert_close(filtered, _direct(volume, envelope, 2.0))
    assert _read_mrc(tmp_path / "out.mrc")[1] == pytest.approx(2.0)


######################
### Config checks ###
######################


@pytest.mark.parametrize("frame_weights", [None, "default", "table"])
def test_config_matched_valid(frame_weights, table_json):
    if frame_weights == "table":
        frame_weights = str(table_json)
    config = SimulatorConfig(
        dose_filter_modify_signal="matched", frame_weights=frame_weights
    )
    assert config.frame_weights == frame_weights


def test_config_matched_template_envelope_valid(envelope_json):
    # the dose range is not used with a template envelope
    config = SimulatorConfig(
        dose_filter_modify_signal="matched",
        template_envelope=str(envelope_json),
        dose_start=0.0,
        dose_end=0.0,
    )
    assert config.template_envelope == str(envelope_json)


def test_config_yaml_round_trip(tmp_path, table_json):
    config = SimulatorConfig(
        dose_filter_modify_signal="matched", frame_weights=str(table_json)
    )
    config.to_yaml(tmp_path / "config.yaml")
    assert SimulatorConfig.from_yaml(tmp_path / "config.yaml") == config


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # frame weights and envelopes only go with 'matched'
        (
            {"dose_filter_modify_signal": "None", "frame_weights": "default"},
            "only used with dose_filter_modify_signal='matched'",
        ),
        (
            {"dose_filter_modify_signal": "rel_diff", "template_envelope": "ENV"},
            "only used with dose_filter_modify_signal='matched'",
        ),
        # one source for the envelope
        ({"frame_weights": "TABLE", "template_envelope": "ENV"}, "not both"),
        # files must exist and parse
        ({"frame_weights": "defualt"}, "Invalid frame weights"),
        ({"frame_weights": "MISSING"}, "Invalid frame weights"),
        ({"template_envelope": "MISSING"}, "Invalid template envelope"),
        ({"frame_weights": "BAD"}, "has no 'exposure_start' field"),
        ({"template_envelope": "BAD"}, "has no 'frame_weights' field"),
        # an envelope file that would not be applied
        (
            {"frame_weights": "default", "apply_dose_weighting": False},
            "need apply_dose_weighting=True",
        ),
        # the G&G critical exposure is not used by 'matched'
        ({"crit_exposure_bfactor": 30.0}, "leave it at -1"),
        # a dose range is needed without a template envelope
        ({"dose_start": 10.0, "dose_end": 10.0}, "dose_end > dose_start"),
    ],
)
def test_config_matched_invalid(tmp_path, table_json, envelope_json, kwargs, match):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"schema": "nothing useful"}))
    files = {
        "TABLE": str(table_json),
        "ENV": str(envelope_json),
        "MISSING": str(tmp_path / "missing.json"),
        "BAD": str(bad_json),
    }
    kwargs = {
        "dose_filter_modify_signal": "matched",
        **{k: files.get(v, v) if isinstance(v, str) else v for k, v in kwargs.items()},
    }
    with pytest.raises(ValidationError, match=match):
        SimulatorConfig(**kwargs)
