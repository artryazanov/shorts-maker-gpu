import sys
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401
from shorts_maker.core.processor import VideoProcessor
from shorts_maker.utils.scenes import combine_scenes, split_overlong_scenes
from shorts_maker.config import ProcessingConfig  # noqa: E402
from shorts_maker.utils.scenes import _SecondsTime  # noqa: E402

def test_combine_scenes_empty():
    config = ProcessingConfig()
    assert combine_scenes([], config) == []

def test_combine_scenes_basic():
    config = ProcessingConfig(min_short_length=2, max_short_length=2, max_combined_scene_length=10)
    
    # Create two scenes that should be combined because they are "small"
    # Wait, the logic combines adjacent runs of "small" ones if less than middle_length/min_length
    s1 = [_SecondsTime(0.0), _SecondsTime(1.0)] # small
    s2 = [_SecondsTime(1.0), _SecondsTime(2.0)] # small
    
    # if both are small, run continues. 
    res = combine_scenes([s1, s2], config)
    # They should be merged into one scene [0.0, 2.0] if final duration >= threshold
    # final_duration = 2.0. threshold = min_short_length = 2. It fits.
    assert len(res) == 1
    assert res[0][0].get_seconds() == 0.0
    assert res[0][1].get_seconds() == 2.0

def test_split_overlong_scenes():
    config = ProcessingConfig(max_short_length=10)
    
    # 50 seconds > 4 * 10 (40) -> splits into 50 / (2*10) = 50 / 20 = 2 parts?
    # n = math.floor(50 / 20) = 2. parts will be 25s each.
    s1 = [_SecondsTime(0.0), _SecondsTime(50.0)]
    
    res = split_overlong_scenes([s1], config)
    assert len(res) == 2
    assert res[0][0].get_seconds() == 0.0
    assert res[0][1].get_seconds() == 25.0
    assert res[1][0].get_seconds() == 25.0
    assert res[1][1].get_seconds() == 50.0

@mock.patch("shorts_maker.core.processor.detect_video_scenes_gpu")
@mock.patch("shorts_maker.core.processor.compute_audio_action_profile")
@mock.patch("shorts_maker.core.processor.compute_video_action_profile")
@mock.patch("shorts_maker.core.processor.render_video_gpu_isolated")
@mock.patch("shorts_maker.core.processor.get_render_params")
def test_process_video(mock_get_params, mock_render, mock_video_action, mock_audio_action, mock_detect, tmp_path):
    config = ProcessingConfig()
    
    s1 = (_SecondsTime(0.0), _SecondsTime(5.0))
    mock_detect.return_value = [s1]
    
    mock_audio_action.return_value = (np.array([0.0, 5.0]), np.array([1.0, 1.0]))
    mock_video_action.return_value = (np.array([0.0, 5.0]), np.array([1.0, 1.0]))
    
    mock_get_params.return_value = mock.MagicMock()
    
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.touch() # Create dummy file
    
    # Mock video probe
    with mock.patch("shorts_maker.utils.scenes.nvc.PyFFmpegDemuxer") as mock_dmx:
        mock_dmx_instance = mock.MagicMock()
        mock_dmx_instance.Numframes.return_value = 150
        mock_dmx_instance.Framerate.return_value = 30.0
        mock_dmx.return_value = mock_dmx_instance
        
        VideoProcessor(config).process_video(dummy_vid, tmp_path)
        
    mock_render.assert_called_once()
    mock_get_params.assert_called_once()

@mock.patch("shorts_maker.core.processor.render_video_gpu_isolated")
@mock.patch("shorts_maker.core.processor.get_render_params")
def test_process_video_no_scenes(mock_get_params, mock_render, tmp_path):
    """Test the fallback branch where no scenes are detected."""
    config = ProcessingConfig()
    
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.touch()

    with mock.patch("shorts_maker.core.processor.detect_video_scenes_gpu", return_value=[]), \
         mock.patch("shorts_maker.core.processor.compute_audio_action_profile", return_value=(np.array([]), np.array([]))), \
         mock.patch("shorts_maker.core.processor.compute_video_action_profile", return_value=(np.array([]), np.array([]))), \
         mock.patch("shorts_maker.core.processor.nvc.PyFFmpegDemuxer") as mock_dmx:
        
        mock_dmx_instance = mock.MagicMock()
        mock_dmx_instance.Numframes.return_value = 3000
        mock_dmx_instance.Framerate.return_value = 30.0
        mock_dmx.return_value = mock_dmx_instance
        
        VideoProcessor(config).process_video(dummy_vid, tmp_path)
    
    # Should randomly sample a clip and render
    mock_render.assert_called_once()
    mock_get_params.assert_called_once()

@mock.patch("shorts_maker.core.processor.render_video_gpu_isolated")
@mock.patch("shorts_maker.core.processor.get_render_params")
def test_process_video_strategies(mock_get_params, mock_render, tmp_path):
    """Test the padding/smart crop strategies when scenes are present."""
    config = ProcessingConfig(max_short_length=15.0, min_short_length=5.0)
    
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.touch()

    # Small scene that fits entirely
    s1 = [_SecondsTime(0.0), _SecondsTime(5.0)]
    # Big scene that is too long
    s2 = [_SecondsTime(20.0), _SecondsTime(40.0)]

    mock_scenes = [s1, s2]

    with mock.patch("shorts_maker.core.processor.detect_video_scenes_gpu", return_value=[]), \
         mock.patch("shorts_maker.core.processor.compute_audio_action_profile", return_value=(np.array([0.0, 50.0]), np.array([1.0, 1.0]))), \
         mock.patch("shorts_maker.core.processor.compute_video_action_profile", return_value=(np.array([0.0, 50.0]), np.array([1.0, 1.0]))), \
         mock.patch("shorts_maker.core.processor.combine_scenes", return_value=mock_scenes), \
         mock.patch("shorts_maker.core.processor.split_overlong_scenes", return_value=mock_scenes), \
         mock.patch("shorts_maker.core.processor.best_action_window_start", return_value=20.0), \
         mock.patch("shorts_maker.core.processor.find_smart_end_point", return_value=30.0), \
         mock.patch("shorts_maker.core.processor.nvc.PyFFmpegDemuxer") as mock_dmx:
        
        mock_dmx_instance = mock.MagicMock()
        mock_dmx_instance.Numframes.return_value = 1500
        mock_dmx_instance.Framerate.return_value = 30.0
        mock_dmx.return_value = mock_dmx_instance
        
        VideoProcessor(config).process_video(dummy_vid, tmp_path)
    
    assert mock_render.call_count == 2
    assert mock_get_params.call_count == 2
    
    # Check the first call parameters (Strategy 1)
    # The padding algorithm adds 1.5s padding to the 5.0s scene = 6.5 duration
    first_call_args = mock_get_params.call_args_list[0][0]
    assert first_call_args[1] == 0.0 # final_start
    assert first_call_args[2] == 6.5 # final_duration
    
    # Check the second call parameters (Strategy 2)
    second_call_args = mock_get_params.call_args_list[1][0]
    assert second_call_args[1] == 20.0
    assert second_call_args[2] == 10.0 # final_end (30) - start (20) = 10
