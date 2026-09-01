"""Tests for the optional teamtomo potential backend."""

from pathlib import Path

import pytest
import torch

from ttsim3d.models import Simulator, SimulatorConfig


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


@pytest.mark.parametrize("backend", ["ttsim3d", "teamtomo"])
@pytest.mark.parametrize("bonded", [False, True])
def test_potential_backends_run(tmp_path: Path, backend: str, bonded: bool):
    pdb_path = tmp_path / "mini.pdb"
    _write_mini_pdb(pdb_path)

    config = SimulatorConfig(
        apply_dose_weighting=False,
        apply_dqe=False,
        store_volume=False,
        upsampling=1,
        use_bonded_scattering_factors=bonded,
        potential_backend=backend,  # type: ignore[arg-type]
    )
    sim = Simulator(
        pdb_filepath=pdb_path,
        pixel_spacing=2.0,
        volume_shape=(16, 16, 16),
        simulator_config=config,
        remove_hydrogens=True,
    )

    volume, spacing = sim.run(device="cpu")

    assert volume.shape == (16, 16, 16)
    assert volume.dtype == torch.float32
    assert spacing == pytest.approx(2.0)
    assert torch.isfinite(volume).all()
