import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401
from shorts_maker.utils.scenes import _SecondsTime, _best_window_single, best_action_window_start  # noqa: E402

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


def test_best_window_single_edge_cases():
    scene = (_SecondsTime(float('inf')), _SecondsTime(10.0))
    assert _best_window_single(scene, 2.0, np.array([]), np.array([])) == float('inf')
    
    scene = (_SecondsTime(10.0), _SecondsTime(5.0))
    assert _best_window_single(scene, 2.0, np.array([]), np.array([])) == 10.0

    scene = (_SecondsTime(0.0), _SecondsTime(10.0))
    # max_allowed_start <= start_sec (window = 20)
    assert _best_window_single(scene, 20.0, np.array([]), np.array([])) == 0.0

    # no data in mask
    assert _best_window_single(scene, 2.0, np.array([-5.0, -1.0]), np.array([0, 0])) == 0.0

    # len < 2
    assert _best_window_single(scene, 2.0, np.array([5.0]), np.array([0])) == 0.0

    # dt <= 0 (same times)
    assert _best_window_single(scene, 2.0, np.array([5.0, 5.0]), np.array([0, 0])) == 0.0

    # len(s_seg) < n_win
    times = np.array([1.0, 1.1, 1.2])
    score = np.array([0, 0, 0])
    # window_length = 5.0 -> n_win = 5.0 / 0.1 = 50
    assert _best_window_single(scene, 5.0, times, score) == 0.0

def test_best_action_window_start_edge_cases():
    scene = (_SecondsTime(float('inf')), _SecondsTime(10.0))
    assert best_action_window_start(scene, 2.0, np.array([1.0, 2.0]), np.array([0, 0]), np.array([1.0, 2.0]), np.array([0, 0])) == float('inf')

    scene = (_SecondsTime(0.0), _SecondsTime(10.0))
    # no audio data in mask
    assert best_action_window_start(scene, 2.0, np.array([-5.0, -1.0]), np.array([0, 0]), np.array([1.0, 2.0]), np.array([0, 0])) == 1.0 # falls back to video
    
    # len audio < 2
    assert best_action_window_start(scene, 2.0, np.array([5.0]), np.array([0]), np.array([1.0, 2.0]), np.array([0, 0])) == 1.0 # falls back to video
    
    # dt <= 0
    assert best_action_window_start(scene, 2.0, np.array([5.0, 5.0]), np.array([0, 0]), np.array([1.0, 2.0]), np.array([0, 0])) == 0.0
    
    # window > duration
    assert best_action_window_start(scene, 20.0, np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), np.array([1.0, 2.0]), np.array([0, 0])) == 0.0
    
    # len < n_win
    times = np.array([1.0, 1.1, 1.2])
    score = np.array([0, 0, 0])
    assert best_action_window_start(scene, 5.0, times, score, np.array([1.0, 2.0]), np.array([0, 0])) == 0.0

