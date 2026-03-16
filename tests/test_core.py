import sys
from pathlib import Path
from unittest import mock
import numpy as np

import tests.mock_gpu
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shorts
from shorts import ProcessingConfig, _SecondsTime

def test_combine_scenes_empty():
    config = ProcessingConfig()
    assert shorts.combine_scenes([], config) == []

def test_combine_scenes_basic():
    config = ProcessingConfig(min_short_length=2, max_short_length=2, max_combined_scene_length=10)
    
    # Create two scenes that should be combined because they are "small"
    # Wait, the logic combines adjacent runs of "small" ones if less than middle_length/min_length
    s1 = [_SecondsTime(0.0), _SecondsTime(1.0)] # small
    s2 = [_SecondsTime(1.0), _SecondsTime(2.0)] # small
    
    # if both are small, run continues. 
    res = shorts.combine_scenes([s1, s2], config)
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
    
    res = shorts.split_overlong_scenes([s1], config)
    assert len(res) == 2
    assert res[0][0].get_seconds() == 0.0
    assert res[0][1].get_seconds() == 25.0
    assert res[1][0].get_seconds() == 25.0
    assert res[1][1].get_seconds() == 50.0

@mock.patch("shorts.detect_video_scenes_gpu")
@mock.patch("shorts.compute_audio_action_profile")
@mock.patch("shorts.compute_video_action_profile")
@mock.patch("shorts.render_video_gpu_isolated")
@mock.patch("shorts.get_render_params")
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
    with mock.patch("shorts.VideoReader") as mock_vr:
        mock_vr_instance = mock.MagicMock()
        mock_vr_instance.__len__.return_value = 150
        mock_vr_instance.get_avg_fps.return_value = 30.0
        mock_vr.return_value = mock_vr_instance
        
        shorts.process_video(dummy_vid, config, tmp_path)
        
    mock_render.assert_called_once()
    mock_get_params.assert_called_once()

@mock.patch("shorts.render_video_gpu_isolated")
@mock.patch("shorts.get_render_params")
def test_process_video_no_scenes(mock_get_params, mock_render, tmp_path):
    """Test the fallback branch where no scenes are detected."""
    config = ProcessingConfig()
    
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.touch()

    with mock.patch("shorts.detect_video_scenes_gpu", return_value=[]), \
         mock.patch("shorts.compute_audio_action_profile", return_value=(np.array([]), np.array([]))), \
         mock.patch("shorts.compute_video_action_profile", return_value=(np.array([]), np.array([]))), \
         mock.patch("shorts.VideoReader") as mock_vr:
        
        mock_vr_instance = mock.MagicMock()
        mock_vr_instance.__len__.return_value = 3000
        mock_vr_instance.get_avg_fps.return_value = 30.0
        mock_vr.return_value = mock_vr_instance
        
        shorts.process_video(dummy_vid, config, tmp_path)
    
    # Should randomly sample a clip and render
    mock_render.assert_called_once()
    mock_get_params.assert_called_once()
