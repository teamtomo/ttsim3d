"""Simple run script."""

import click
from pydanclick import from_pydantic

from ttsim3d.models import Simulator


# TODO: Add brief description for each argument in the help message
# TODO: Test the script for input validation
@click.command()
@from_pydantic(
    Simulator,
    exclude=[  # These are large, non-serializable tensors
        "atom_positions_zyx",
        "atom_identities",
        "atom_b_factors",
        "volume",
    ],
    docstring_style="numpy",
    parse_docstring=True,
    rename={"simulator_config": "--config"},
)  # type: ignore
@click.option(
    "--mrc-filepath",
    type=click.Path(),
    required=True,
    help="The file path to save the MRC file.",
)
@click.option(
    "--gpu-ids",
    type=list[int],
    multiple=True,
    help="A list of GPU IDs to use for the simulation.",
)
def main(params: Simulator, mrc_filepath: str, gpu_ids: tuple[int, ...]) -> None:
    """A test function to run the simulate3d function from the ttsim3d package."""
    simulator = Simulator(**params.dict())
    simulator.export_to_mrc(
        mrc_filepath=mrc_filepath, gpu_ids=list(gpu_ids) if gpu_ids else None
    )


if __name__ == "__main__":
    main()
