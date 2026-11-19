import mrcfile
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from ttsim3d import Simulator
from ttsim3d.cli import main


def _read(path):
    with mrcfile.open(path) as mrc:
        return mrc.data.copy(), float(mrc.voxel_size.x)


def test_cli_matches_simulator(tmp_path, dipeptide_pdb):
    output = tmp_path / "out.mrc"
    result = CliRunner().invoke(
        main,
        [
            "--structure", str(dipeptide_pdb),
            "--pixel-spacing", "1.0",
            "--volume-shape", "24", "24", "24",
            "--mtf", "k2_300kv",
            "-o", str(output),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    data, voxel_size = _read(output)
    assert data.shape == (24, 24, 24)
    assert voxel_size == pytest.approx(1.0)
    expected = Simulator(
        structure={"path": dipeptide_pdb},
        pixel_spacing=1.0,
        volume_shape=(24, 24, 24),
        mtf="k2_300kv",
    ).run()
    np.testing.assert_allclose(data, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_cli_config_with_overrides(tmp_path, dipeptide_pdb):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "structure": {"path": str(dipeptide_pdb)},
                "pixel_spacing": 1.0,
                "volume_shape": [24, 24, 24],
                "dose_filter": {"end": 50.0},
                "mtf": "k2_300kv",
            }
        )
    )
    output = tmp_path / "out.mrc"
    result = CliRunner().invoke(
        main,
        [
            "--config", str(config),
            "--pixel-spacing", "1.5",
            "--no-dose-filter",
            "--mtf", "none",
            "-o", str(output),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    data, voxel_size = _read(output)
    assert voxel_size == pytest.approx(1.5)
    expected = Simulator(
        structure={"path": dipeptide_pdb},
        pixel_spacing=1.5,
        volume_shape=(24, 24, 24),
        dose_filter=None,
    ).run()
    np.testing.assert_allclose(data, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_cli_dose_options_enable_filter(tmp_path, dipeptide_pdb):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "structure": {"path": str(dipeptide_pdb)},
                "pixel_spacing": 1.0,
                "volume_shape": [24, 24, 24],
                "dose_filter": None,
            }
        )
    )
    output = tmp_path / "out.mrc"
    args = ["--config", str(config), "--dose-end", "60", "-o", str(output)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output

    data, _ = _read(output)
    expected = Simulator(
        structure={"path": dipeptide_pdb},
        pixel_spacing=1.0,
        volume_shape=(24, 24, 24),
        dose_filter={"end": 60.0},
    ).run()
    np.testing.assert_allclose(data, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_cli_reports_invalid_settings(tmp_path, dipeptide_pdb):
    result = CliRunner().invoke(
        main,
        ["--structure", str(dipeptide_pdb), "-o", str(tmp_path / "out.mrc")],
    )
    assert result.exit_code == 2
    assert "pixel_spacing" in result.output
