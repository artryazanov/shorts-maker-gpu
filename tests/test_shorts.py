import sys
from unittest.mock import MagicMock
import numpy as np
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, cupy, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.



# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import shorts AFTER mocking
import shorts  # noqa: E402
from shorts import (  # noqa: E402
    blur_gpu,
    combine_scenes,
    select_background_resolution,
    ProcessingConfig,
    compute_video_action_profile,
    _SecondsTime,
)


# Helper to create scene tuples
def make_scene(start: float, end: float):
    return (_SecondsTime(start), _SecondsTime(end))


def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2100) == (2160, 3840)


def test_blur_gpu_uses_cupy():
    # Verify blur_gpu calls cupy/cupyx functions
    # Input is a torch tensor mock
    image_tensor = MagicMock()
    image_tensor.is_contiguous.return_value = True

    # Configure mock return for gaussian_filter
    # It returns a cupy array mock
    mock_cupy_array = MagicMock()
    shorts.cupyx.scipy.ndimage.gaussian_filter.return_value = mock_cupy_array

    # Return mock torch tensor
    shorts.torch.utils.dlpack.from_dlpack.return_value = MagicMock()

    blur_gpu(image_tensor)

    shorts.torch.to_dlpack.assert_called_with(image_tensor)
    shorts.cp.from_dlpack.assert_called()
    shorts.cupyx.scipy.ndimage.gaussian_filter.assert_called()
    shorts.torch.utils.dlpack.from_dlpack.assert_called()


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
    assert len(combined) == 1
    start, end = combined[0]
    assert start.get_seconds() == 5
    assert end.get_seconds() == 13

# render_video (legacy) has been removed.
# render_video_gpu logic is verified via mocks in separate flows or implicitly here if we add such tests.


def test_compute_video_action_profile_sequential():
    """Verify that compute_video_action_profile reads sequentially (batch-by-batch) and subsamples."""
    
    # 1. Setup Mock VideoReader
    mock_vr = MagicMock()
    # Let's say video has 1000 frames, 30 fps
    mock_vr.__len__.return_value = 1000
    mock_vr.get_avg_fps.return_value = 30.0

    # Configure __getitem__ for metadata probe (vr_cpu[0].shape)
    mock_frame = MagicMock()
    mock_frame.shape = (720, 1280, 3)
    mock_vr.__getitem__.return_value = mock_frame

    # Configure get_batch to return a FakeTensor
    def side_effect_get_batch(indices):
        from tests.mock_gpu import FakeTensor
        count = len(indices)
        return FakeTensor(shape=(count, 64, 64, 3), numel=count*64*64*3)

    mock_vr.get_batch.side_effect = side_effect_get_batch

    shorts.VideoReader.return_value = mock_vr

    times, scores = compute_video_action_profile(Path("dummy.mp4"), fps=6)

    assert shorts.VideoReader.called
    assert mock_vr.get_batch.called

    calls = mock_vr.get_batch.call_args_list
    assert len(calls) > 0
    
    first_call_args = calls[0].args[0]
    assert list(first_call_args) == list(range(0, 16))
    
    assert isinstance(times, np.ndarray) or (times == [])
