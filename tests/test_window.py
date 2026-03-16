import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shorts import _SecondsTime, _best_window_single, best_action_window_start

def test_best_window_single():
    scene = (_SecondsTime(0.0), _SecondsTime(10.0))
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Spike at second 5
    score = np.array([0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
    
    # window_length = 2.0. Expected to start around 4 or 5.
    # 5 is at index 5. csum logic will find best_start.
    best_start = _best_window_single(scene, window_length=2.0, times=times, score=score)
    # The sum of window [4,5], [5,6] -> max is around 5.0. 
    # Because of the logic, it should capture the spike at 5.0.
    assert best_start > 0.0

def test_best_window_single_invalid():
    # Very short scene
    scene = (_SecondsTime(0.0), _SecondsTime(1.0))
    times = np.array([0.0, 1.0])
    score = np.array([0, 0])
    best_start = _best_window_single(scene, window_length=5.0, times=times, score=score)
    assert best_start == 0.0

def test_best_action_window_start_audio_video():
    scene = (_SecondsTime(0.0), _SecondsTime(10.0))
    a_times = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    a_score = np.array([0, 0, 10, 0, 0, 0]) # Spike at 4.0
    
    v_times = np.array([0.0, 3.0, 6.0, 9.0])
    v_score = np.array([0, 10, 0, 0]) # Spike at 3.0
    
    # Combined action will peak between 3 and 4
    best_start = best_action_window_start(
        scene, 2.0, a_times, a_score, v_times, v_score, w_audio=0.5, w_video=0.5
    )
    assert best_start > 0.0

def test_best_action_window_start_no_video():
    scene = (_SecondsTime(0.0), _SecondsTime(10.0))
    a_times = np.array([0.0, 5.0, 10.0])
    a_score = np.array([0, 10, 0])
    
    # Pass None for video
    best_start = best_action_window_start(
        scene, 2.0, a_times, a_score, None, None
    )
    assert best_start > 0.0
