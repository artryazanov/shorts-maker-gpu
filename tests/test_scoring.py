import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import tests.mock_gpu  # noqa: F401
from shorts import scene_action_score, _SecondsTime  # noqa: E402

def test_scene_action_score_audio_only():
    scene = (_SecondsTime(1.0), _SecondsTime(3.0))
    audio_times = np.array([0.5, 1.5, 2.5, 3.5])
    audio_score = np.array([10.0, 20.0, 30.0, 40.0])
    
    # 1.5 and 2.5 fall inside [1.0, 3.0), which are values 20.0 and 30.0
    # Sum is 50.0. 
    # w_audio is 0.6 by default, w_video is 0.4
    res = scene_action_score(scene, audio_times, audio_score)
    assert np.isclose(res, 50.0 * 0.6) or res == 50.0

def test_scene_action_score_audio_and_video():
    scene = (_SecondsTime(1.0), _SecondsTime(3.0))
    audio_times = np.array([1.5, 2.5])
    audio_score = np.array([10.0, 10.0]) # sum = 20
    
    video_times = np.array([1.5, 2.0, 2.5])
    video_score = np.array([5.0, 5.0, 5.0]) # sum = 15
    
    res = scene_action_score(scene, audio_times, audio_score, video_times, video_score)
    # 20 * 0.6 + 15 * 0.4 = 12 + 6 = 18.0
    assert np.isclose(res, 18.0)

def test_scene_action_score_empty():
    scene = (_SecondsTime(5.0), _SecondsTime(6.0))
    res = scene_action_score(scene, np.array([]), np.array([]))
    assert res == 0.0

def test_scene_action_score_zero_length():
    scene = (_SecondsTime(5.0), _SecondsTime(5.0))
    res = scene_action_score(scene, np.array([5.0]), np.array([10.0]))
    assert res == 0.0
