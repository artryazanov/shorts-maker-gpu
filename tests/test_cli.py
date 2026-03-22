import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from shorts_maker.cli import app

runner = CliRunner()

def test_cli_process_no_input_dir(tmp_path):
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "nonexistent"
    
    result = runner.invoke(app, ["--input-dir", str(input_dir), "--output-dir", str(output_dir)])
    assert result.exit_code == 1

def test_cli_process_success(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    (input_dir / "video1.mp4").touch()
    (input_dir / "video2.mkv").touch()
    (input_dir / "not_video.txt").touch()

    with patch("shorts_maker.cli.VideoProcessor") as mock_vp_class:
        mock_processor = MagicMock()
        mock_vp_class.return_value = mock_processor
        
        result = runner.invoke(
            app, 
            ["--input-dir", str(input_dir), "--output-dir", str(output_dir), "--scene-limit", "5"]
        )
        
        assert result.exit_code == 0
        assert mock_processor.process_video.call_count == 2

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
