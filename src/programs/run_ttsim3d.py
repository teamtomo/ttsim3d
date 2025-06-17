"""Simple run script."""

from typing import Union

import click

from ttsim3d.models import Simulator, SimulatorConfig


@click.command()
@click.option(
    "--pdb-filepath",
    type=click.Path(),
    required=True,
    help="The path to the PDB file containing the atomic structure to simulate.",
)
@click.option(
    "--mrc-filepath",
    type=click.Path(),
    required=True,
    help="File path to save simulated volume.",
)
@click.option(
    "--pixel-spacing",
    type=float,
    help=(
        "The pixel spacing of the simulated volume in units of Angstroms. "
        "Must be greater than 0."
    ),
)
@click.option(
    "--volume-shape",
    type=(int, int, int),
    help="The shape of the simulated volume in pixels.",
)
@click.option(
    "--voltage",
    type=float,
    default=300.0,
    help="The voltage of the microscope in kV. Default is 300 kV.",
)
@click.option(
    "--upsampling",
    type=int,
    default=-1,
    help=(
        "The upsampling factor to apply to the simulation. The default is -1 "
        "and corresponds to automatic calculation of the upsampling factor."
    ),
)
@click.option(
    "--b-factor-scaling",
    type=float,
    default=1.0,
    help=(
        "The scaling factor to apply to the B-factors of the atoms in the "
        "pdb file. The default is 1.0."
    ),
)
@click.option(
    "--additional-b-factor",
    type=float,
    default=0.0,
    help=(
        "Additional B-factor to apply to the atoms in the pdb file. The "
        "default is 0.0."
    ),
)
@click.option(
    "--apply-dose-weighting",
    type=bool,
    default=True,
    help="If True, apply dose weighting to the simulation. Default is True.",
)
@click.option(
    "--crit-exposure-bfactor",
    type=float,
    default=-1.0,
    help=(
        "B-factor to use in critical exposure calculations. The default is -1 "
        "and corresponds to the fitted critical exposure function in Grant "
        "and Grigorieff, 2015."
    ),
)
@click.option(
    "--dose-filter-modify-signal",
    type=click.Choice(["None", "sqrt", "rel_diff"]),
    default="None",
    help=(
        "Signal modification to apply to the dose filter. Currently supports "
        "'None', 'sqrt', and 'rel_diff'."
    ),
)
@click.option(
    "--dose-start",
    type=float,
    default=0.0,
    help="The starting dose in e/A^2. Default is 0.0.",
)
@click.option(
    "--dose-end",
    type=float,
    default=30.0,
    help="The ending dose in e/A^2. Default is 30.0.",
)
@click.option(
    "--apply-dqe",
    is_flag=True,
    help="If True, apply a DQE filter to the simulation.",
)
@click.option(
    "--mtf-reference",
    type=str,
    default="k2_300kV",
    help=(
        "Path to the modulation transfer function (MTF) reference star "
        "file, or one of the known MTF reference files. "
        "Default is 'k2_300kV'."
    ),
)
@click.option(
    "--gpu-ids",
    type=Union[int, list[int], str, list[str]],
    multiple=True,
    help=(
        "A single integers (e.g. '0') or string (e.g. 'cuda:0') specifying which GPU "
        "device(s) to use. Also supports lists of integers or strings, but underlying "
        "computation only runs on a single GPU. 'cpu' will run on the CPU."
    ),
)
@click.option(
    "--atom-batch-size",
    type=int,
    default=16384,
    help=(
        "The number of atoms to process (simulate the scattering potentials of) at a "
        "single time. This is partially controls the memory usage. If -1, the batch "
        "size calculated automatically. Default is 16384 (2^14)."
    ),
)
def run_simulation_cli(
    pdb_filepath: str,
    mrc_filepath: str,
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    voltage: float,
    upsampling: int,
    b_factor_scaling: float,
    additional_b_factor: float,
    apply_dose_weighting: bool,
    crit_exposure_bfactor: float,
    dose_filter_modify_signal: str,
    dose_start: float,
    dose_end: float,
    apply_dqe: bool,
    mtf_reference: str,
    device: Union[int, list[int], str, list[str]],
    atom_batch_size: int,
) -> None:
    """Run a structure simulation through the CLI."""
    simulator_config = SimulatorConfig(
        voltage=voltage,
        apply_dose_weighting=apply_dose_weighting,
        crit_exposure_bfactor=crit_exposure_bfactor,
        dose_filter_modify_signal=dose_filter_modify_signal,
        dose_start=dose_start,
        dose_end=dose_end,
        apply_dqe=apply_dqe,
        mtf_reference=mtf_reference,
        upsampling=upsampling,
        atom_batch_size=atom_batch_size,
    )
    simulator = Simulator(
        pixel_spacing=pixel_spacing,
        volume_shape=volume_shape,
        pdb_filepath=pdb_filepath,
        b_factor_scaling=b_factor_scaling,
        additional_b_factor=additional_b_factor,
        simulator_config=simulator_config,
    )
    simulator.export_to_mrc(device=device, mrc_filepath=mrc_filepath)


if __name__ == "__main__":
    run_simulation_cli()
