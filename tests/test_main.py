import sys
from pathlib import Path

# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shorts_maker.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    """Verify that the CLI helps text displays properly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Processes long gameplay videos" in result.stdout
