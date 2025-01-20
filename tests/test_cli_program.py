"""Tests that the CLI program is available (does not check outputs)"""

from click.testing import CliRunner
from src.programs.run_ttsim3d import run_simulation_cli


def test_cli_help():
    """Asserts help message exits with status code 0 (successful)."""
    runner = CliRunner()
    result = runner.invoke(run_simulation_cli, ["--help"])
    assert result.exit_code == 0
