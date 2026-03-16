import sys
from unittest.mock import MagicMock
from pathlib import Path
import os
import pytest

import tests.mock_gpu

# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shorts

def test_get_env_int(monkeypatch):
    monkeypatch.setenv("TEST_INT", "42")
    assert shorts._get_env_int("TEST_INT", 10) == 42
    
    monkeypatch.setenv("TEST_INT_INVALID", "abc")
    assert shorts._get_env_int("TEST_INT_INVALID", 10) == 10
    
    monkeypatch.delenv("TEST_INT_MISSING", raising=False)
    assert shorts._get_env_int("TEST_INT_MISSING", 15) == 15

def test_get_env_float(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "3.14")
    assert shorts._get_env_float("TEST_FLOAT", 1.0) == 3.14

    monkeypatch.setenv("TEST_FLOAT_INVALID", "xyz")
    assert shorts._get_env_float("TEST_FLOAT_INVALID", 2.5) == 2.5

    monkeypatch.delenv("TEST_FLOAT_MISSING", raising=False)
    assert shorts._get_env_float("TEST_FLOAT_MISSING", 4.0) == 4.0

def test_processing_config():
    config = shorts.ProcessingConfig(min_short_length=20, max_short_length=60)
    assert config.middle_short_length == 40.0

def test_seconds_time():
    st = shorts._SecondsTime(1.5)
    assert st.get_seconds() == 1.5
    assert st.get_timecode() == "1.50"
    assert st.get_frames() == 45

def test_log_memory_usage(caplog):
    import logging
    caplog.set_level(logging.INFO)
    shorts.log_memory_usage("test_tag")
    assert "test_tag" in caplog.text
    assert "Memory:" in caplog.text

def test_render_params():
    rp = shorts.RenderParams(
        source_path=Path("dummy.mp4"),
        start_time=1.0,
        duration=10.0,
        output_width=1080,
        output_height=1920,
        crop_x=0, crop_y=0, crop_w=1080, crop_h=1080,
        bg_width=1080, bg_height=1920,
        is_vertical_bg=True
    )
    assert rp.output_width == 1080
    assert rp.is_vertical_bg is True
