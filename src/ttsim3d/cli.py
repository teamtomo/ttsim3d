"""Command line interface: ``ttsim3d-cli``."""

from __future__ import annotations

from typing import Any

import click
import yaml
from pydantic import ValidationError

from ttsim3d.config import Simulator


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML file with Simulator settings. Command line options override it.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output MRC file.",
)
@click.option(
    "--structure",
    "structure_path",
    type=click.Path(exists=True, dir_okay=False),
    help="PDB or mmCIF file to simulate.",
)
@click.option("--pixel-spacing", type=float, help="Output pixel spacing in Å.")
@click.option(
    "--volume-shape", type=(int, int, int), help="Output volume shape (Z Y X)."
)
@click.option(
    "--center/--no-center",
    default=None,
    help="Center the atoms on the volume (default: center).",
)
@click.option(
    "--remove-hydrogens/--keep-hydrogens",
    default=None,
    help="Drop hydrogen atoms (default: remove).",
)
@click.option("--b-factor-scale", type=float, help="Multiply B-factors by this.")
@click.option("--b-factor-offset", type=float, help="Add this B-factor (Å²).")
@click.option(
    "--scattering-factors",
    type=click.Choice(["peng_elemental", "peng_bonded"]),
    help="Scattering-factor model (default: peng_elemental).",
)
@click.option("--upsampling", type=click.IntRange(min=1), help="Default: automatic.")
@click.option("--sublattice-radius", type=float, help="Default: from B-factors.")
@click.option("--batch-size", type=click.IntRange(min=1), help="Atoms per chunk.")
@click.option(
    "--dose-filter/--no-dose-filter",
    default=None,
    help="Apply the cumulative dose filter (default: apply).",
)
@click.option("--dose-start", type=float, help="Exposure start in e⁻/Å².")
@click.option("--dose-end", type=float, help="Exposure end in e⁻/Å².")
@click.option(
    "--critical-bfactor",
    type=float,
    help="Critical-exposure B-factor (default: Grant & Grigorieff 2015).",
)
@click.option(
    "--dose-modify-signal",
    type=click.Choice(["none", "sqrt", "rel_diff"]),
    help="Transformation applied to the dose filter.",
)
@click.option(
    "--mtf",
    help="Bundled detector MTF name or star file path; 'none' disables it.",
)
@click.option(
    "--device", default="cpu", show_default=True, help="e.g. 'cpu' or 'cuda:0'."
)
def main(
    config_path: str | None,
    output: str,
    structure_path: str | None,
    pixel_spacing: float | None,
    volume_shape: tuple[int, int, int] | None,
    center: bool | None,
    remove_hydrogens: bool | None,
    b_factor_scale: float | None,
    b_factor_offset: float | None,
    scattering_factors: str | None,
    upsampling: int | None,
    sublattice_radius: float | None,
    batch_size: int | None,
    dose_filter: bool | None,
    dose_start: float | None,
    dose_end: float | None,
    critical_bfactor: float | None,
    dose_modify_signal: str | None,
    mtf: str | None,
    device: str,
) -> None:
    """Simulate a 3D electrostatic potential (volts) from a PDB/mmCIF file."""
    settings: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}

    _update(settings, pixel_spacing=pixel_spacing, volume_shape=volume_shape)
    _update(
        settings.setdefault("structure", {}),
        path=structure_path,
        center=center,
        remove_hydrogens=remove_hydrogens,
        b_factor_scale=b_factor_scale,
        b_factor_offset=b_factor_offset,
    )
    _update(
        settings.setdefault("potential", {}),
        scattering_factors=scattering_factors,
        upsampling=upsampling,
        sublattice_radius=sublattice_radius,
        batch_size=batch_size,
    )

    if dose_filter is False:
        settings["dose_filter"] = None
    else:
        dose_settings = settings.get("dose_filter") or {}
        _update(
            dose_settings,
            start=dose_start,
            end=dose_end,
            critical_bfactor=critical_bfactor,
            modify_signal=dose_modify_signal,
        )
        if dose_filter or dose_settings or "dose_filter" not in settings:
            settings["dose_filter"] = dose_settings

    if mtf is not None:
        settings["mtf"] = None if mtf.lower() == "none" else mtf

    try:
        simulator = Simulator.model_validate(settings)
    except ValidationError as error:
        raise click.UsageError(str(error)) from error

    simulator.export_to_mrc(output, device=device)


def _update(target: dict[str, Any], **values: Any) -> None:
    """Set the entries of ``values`` that are not None."""
    target.update({key: value for key, value in values.items() if value is not None})


if __name__ == "__main__":
    main()
