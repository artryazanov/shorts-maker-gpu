import sys
from pathlib import Path

# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401
from shorts_maker.config import ProcessingConfig
from shorts_maker.utils.scenes import _SecondsTime
from shorts_maker.io.render import RenderParams
from shorts_maker.io.render import log_memory_usage

def test_processing_config():
    config = ProcessingConfig(min_short_length=20, max_short_length=60)
    assert config.middle_short_length == 40.0

def test_seconds_time():
    st = _SecondsTime(1.5)
    assert st.get_seconds() == 1.5
    assert st.get_timecode() == "1.50"
    assert st.get_frames() == 45

def test_log_memory_usage(caplog):
    import logging
    caplog.set_level(logging.INFO)
    log_memory_usage("test_tag")
    assert "test_tag" in caplog.text
    assert "Memory:" in caplog.text

def test_render_params():
    rp = RenderParams(
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
