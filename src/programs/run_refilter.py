"""Apply a matched template envelope to an existing simulated map."""

import os
from typing import Optional

import click
from click.core import ParameterSource

from ttsim3d.envelope import (
    DEFAULT_FRAME_WEIGHTS,
    load_template_envelope,
    refilter_mrc,
)


@click.command()
@click.option(
    "--input-mrc",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=(
        "Map simulated without dose weighting, e.g. by "
        "'ttsim3d-cli --apply-dose-weighting False'."
    ),
)
@click.option(
    "--output-mrc",
    type=click.Path(dir_okay=False),
    required=True,
    help="File path to save the filtered map. It gets the input's voxel size.",
)
@click.option(
    "--template-envelope",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a template envelope JSON (frame weights plus an optional smooth "
        "correction). Replaces --frame-weights and the dose range."
    ),
)
@click.option(
    "--frame-weights",
    type=str,
    default=None,
    help=(
        "'default' for the default per-frame B-factor curve, or the path to a "
        "frame weight table JSON, both over frames of 1 e/A^2 from --dose-start "
        "to --dose-end. Default is the default curve, unless --template-envelope "
        "is given."
    ),
)
@click.option(
    "--dose-start",
    type=float,
    default=0.0,
    help="The starting dose of the sum in e/A^2. Default is 0.0.",
)
@click.option(
    "--dose-end",
    type=float,
    default=30.0,
    help="The ending dose of the sum in e/A^2. Default is 30.0.",
)
@click.option(
    "--pixel-spacing",
    type=float,
    default=None,
    help="Voxel size of the input map in Angstroms. Default: read from its header.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "Device for the FFTs, e.g. 'cpu' or 'cuda:0' (or '0'). The default is 'cpu'."
    ),
)
def refilter_cli(
    input_mrc: str,
    output_mrc: str,
    template_envelope: Optional[str],
    frame_weights: Optional[str],
    dose_start: float,
    dose_end: float,
    pixel_spacing: Optional[float],
    device: str,
) -> None:
    """Apply a matched template envelope to a map simulated without dose weighting.

    The envelope comes from --template-envelope, or from --frame-weights over
    --dose-start to --dose-end. This is the same filter as
    'ttsim3d-cli --dose-filter-modify-signal matched', so templates can be updated
    for new frame weights without simulating again.
    """
    ctx = click.get_current_context()
    if template_envelope is not None:
        if frame_weights is not None:
            raise click.UsageError(
                "Give --template-envelope or --frame-weights, not both: a template "
                "envelope already holds its frame weights."
            )
        for name in ("dose_start", "dose_end"):
            if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
                raise click.UsageError(
                    f"--{name.replace('_', '-')} is not used with "
                    "--template-envelope, which holds its own frames."
                )
    elif (
        frame_weights is not None
        and frame_weights != DEFAULT_FRAME_WEIGHTS
        and not os.path.isfile(frame_weights)
    ):
        raise click.BadParameter(
            f"'{frame_weights}' is neither '{DEFAULT_FRAME_WEIGHTS}' nor a file.",
            param_hint="'--frame-weights'",
        )

    try:
        envelope = load_template_envelope(
            dose_start=dose_start,
            dose_end=dose_end,
            frame_weights=frame_weights,
            template_envelope=template_envelope,
        )
    except (KeyError, TypeError, ValueError) as err:
        raise click.UsageError(f"Could not build the envelope: {err}") from err

    refilter_mrc(
        input_mrc=input_mrc,
        output_mrc=output_mrc,
        envelope=envelope,
        pixel_spacing=pixel_spacing,
        device=device,
    )
    click.echo(f"Wrote {output_mrc}")


if __name__ == "__main__":
    refilter_cli()
