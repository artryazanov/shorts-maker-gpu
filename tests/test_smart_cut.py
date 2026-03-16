import sys
from unittest.mock import MagicMock
import numpy as np
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, cupy, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.
import tests.mock_gpu



# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import shorts AFTER mocking
from shorts import find_smart_end_point  # noqa: E402


def test_find_smart_end_point_basic():
    """Test basic functionality: finding a local minimum."""
    # Times: 0..10
    # Scores: high at edges, low at 5.0
    times = np.linspace(0, 10, 11)  # 0.0, 1.0, ..., 10.0
    scores = np.abs(times - 5.0)    # 5.0, 4.0, ..., 0.0, ..., 5.0
    
    # Search window covers the minimum
    # max_end = 8.0, window = 5.0 -> search [3.0, 8.0]
    # min at 5.0 is in range
    res = find_smart_end_point(
        start_time=0.0,
        min_end=0.0,
        max_end=8.0,
        times=times,
        scores=scores,
        search_window=5.0
    )
    
    assert res == 5.0


def test_find_smart_end_point_boundary():
    """Test when the best point is at the boundary."""
    times = np.array([10.0, 11.0, 12.0])
    scores = np.array([10.0, 5.0, 1.0])  # Decreasing, best is at end
    
    res = find_smart_end_point(
        start_time=0.0,
        min_end=0.0,
        max_end=12.0,
        times=times,
        scores=scores,
        search_window=5.0
    )
    
    assert res == 12.0


def test_find_smart_end_point_no_data_in_window():
    """Test fallback when no data points exist in the search window."""
    times = np.array([0.0, 1.0])
    scores = np.array([1.0, 1.0])
    
    # Search window [5.0, 7.0] -> disjoint from data [0.0, 1.0]
    res = find_smart_end_point(
        start_time=0.0,
        min_end=0.0,
        max_end=7.0,
        times=times,
        scores=scores,
        search_window=2.0
    )
    
    # Should return max_end default
    assert res == 7.0


def test_find_smart_end_point_constrained_min_end():
    """Test respect for min_end constraint."""
    # times: 10, 11, 12, 13
    # scores: 10, 1, 10, 10
    # Global min at 11.0
    times = np.array([10.0, 11.0, 12.0, 13.0])
    scores = np.array([10.0, 1.0, 10.0, 10.0])
    
    # max_end = 13.0, window = 5.0 -> normally search starts at 8.0
    # BUT min_end = 12.0. So search range becomes [12.0, 13.0]
    # 11.0 (score 1.0) is OUT of range.
    # In range [12.0, 13.0], scores are 10.0(at 12.0) and 10.0(at 13.0).
    # argmin likely picks the first one (0 index in segment) -> 12.0
    
    res = find_smart_end_point(
        start_time=0.0,
        min_end=12.0,
        max_end=13.0,
        times=times,
        scores=scores,
        search_window=5.0
    )
    
    assert res == 12.0
