"""Tests that the CLI program is available (does not check outputs)"""

from click.testing import CliRunner
from src.programs.run_refilter import refilter_cli
from src.programs.run_ttsim3d import run_simulation_cli

from ttsim3d.models import SimulatorConfig


def test_cli_help():
    """Asserts help message exits with status code 0 (successful)."""
    runner = CliRunner()
    result = runner.invoke(run_simulation_cli, ["--help"])
    assert result.exit_code == 0


def test_cli_help_lists_matched_options():
    """The matched dose filter options are on the simulation program."""
    result = CliRunner().invoke(run_simulation_cli, ["--help"])
    assert result.exit_code == 0
    for option in ("matched", "--frame-weights", "--template-envelope"):
        assert option in result.output


def test_refilter_cli_help():
    """Asserts the refilter program's help message exits with status code 0."""
    result = CliRunner().invoke(refilter_cli, ["--help"])
    assert result.exit_code == 0


def test_cli_defaults_are_valid():
    """The CLI's default options make a valid simulation config."""
    # Let click resolve the defaults (p.default is a sentinel for flags and options
    # without a default from click 8.3 on); resilient parsing skips the required
    # file options.
    ctx = run_simulation_cli.make_context("ttsim3d-cli", [], resilient_parsing=True)
    defaults = {
        name: value
        for name, value in ctx.params.items()
        if name in SimulatorConfig.model_fields
    }
    assert "mtf_reference" in defaults
    SimulatorConfig(**defaults)
