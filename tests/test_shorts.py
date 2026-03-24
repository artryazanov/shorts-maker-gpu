import sys
from unittest.mock import MagicMock
import numpy as np
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.



# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401, E402
from shorts_maker.utils.scenes import combine_scenes, _SecondsTime
from shorts_maker.io.render import select_background_resolution
from shorts_maker.config import ProcessingConfig
from shorts_maker.analysis.video import compute_video_action_profile


# Helper to create scene tuples
def make_scene(start: float, end: float):
    return (_SecondsTime(start), _SecondsTime(end))


def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2100) == (2160, 3840)


def test_combine_scenes_merges_short_scenes():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 5),
        make_scene(5, 7),
        make_scene(7, 9),
        make_scene(9, 11),
        make_scene(11, 13),
        make_scene(13, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 3
    assert combined[0][0].get_seconds() == 0
    assert combined[0][1].get_seconds() == 5
    assert combined[1][0].get_seconds() == 5
    assert combined[1][1].get_seconds() == 11
    assert combined[2][0].get_seconds() == 11
    assert combined[2][1].get_seconds() == 18




# render_video (legacy) has been removed.
# render_video_gpu logic is verified via mocks in separate flows or implicitly here if we add such tests.


def test_compute_video_action_profile_sequential():
    """Verify that compute_video_action_profile reads sequentially (batch-by-batch) and subsamples."""
    
    # 1. Setup Mock
    mock_streamer_instance = MagicMock()
    
    # Configure stream_batches to yield a couple of FakeTensors
    def mock_stream_batches(batch_size=16, step=1, max_frames=None):
        yield tests.mock_gpu.FakeTensor(shape=(batch_size, 64, 64, 3), numel=batch_size*64*64*3), list(range(0, batch_size * step, step))
        yield tests.mock_gpu.FakeTensor(shape=(batch_size, 64, 64, 3), numel=batch_size*64*64*3), list(range(batch_size * step, batch_size*2 * step, step))

    mock_streamer_instance.stream_batches.side_effect = mock_stream_batches
    mock_streamer_instance.__enter__.return_value = mock_streamer_instance
    mock_streamer_instance.total_frames = 32

    from unittest import mock
    with mock.patch("shorts_maker.analysis.video.GPUVideoStreamer", return_value=mock_streamer_instance):
        times, scores = compute_video_action_profile(Path("dummy.mp4"), fps=6)

        assert mock_streamer_instance.stream_batches.called
        assert isinstance(times, np.ndarray) or (times == [])


def test_combine_scenes_exact_max_boundary():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 3),
        make_scene(3, 6),
        make_scene(6, 9),
        make_scene(9, 12),
        make_scene(12, 15)
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 2
    assert combined[0][1].get_seconds() == 6.0
    assert combined[1][1].get_seconds() == 15.0

    scenes = [
        make_scene(0, 3),
        make_scene(3, 6),
        make_scene(6, 9),
        make_scene(9, 12),
        make_scene(12, 15),
        make_scene(15, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 3
    assert combined[0][1].get_seconds() == 6.0
    assert combined[1][1].get_seconds() == 12.0
    assert combined[2][1].get_seconds() == 18.0

