import sys
from pathlib import Path
from unittest import mock

import tests.mock_gpu
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shorts
from shorts import detect_video_scenes_gpu

def test_detect_video_scenes_empty_video():
    """Test scene detection when video has 0 frames."""
    mock_vr = mock.MagicMock()
    mock_vr.__len__.return_value = 0
    mock_vr.get_avg_fps.return_value = 30.0
    mock_vr.__getitem__.return_value.shape = (1080, 1920, 3)
    
    shorts.VideoReader = mock.MagicMock(return_value=mock_vr)
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"))
    assert len(scenes) == 0

def test_detect_video_scenes_no_cuts():
    """Test scene detection when no cuts exist."""
    mock_vr = mock.MagicMock()
    mock_vr.__len__.return_value = 60 # 60 frames = 2 seconds at 30fps
    mock_vr.get_avg_fps.return_value = 30.0
    mock_vr.__getitem__.return_value.shape = (1080, 1920, 3)
    
    class FakeFramesTensor:
        def detach(self): return self
        def to(self, _): return self
        def numpy(self):
            import numpy as np
            # Return identical gray frames (no scene changes)
            return np.ones((16, 256, 144, 3), dtype=np.uint8) * 128
            
    mock_vr.get_batch.return_value = FakeFramesTensor()
    
    shorts.VideoReader = mock.MagicMock(return_value=mock_vr)
    
    import numpy as np
    
    cv2_mock = mock.MagicMock()
    def dummy_cvt(img, mode):
        return np.ones_like(img) * 10
    cv2_mock.cvtColor.side_effect = dummy_cvt
    cv2_mock.COLOR_BGR2HSV = 40
    
    def dummy_split(hsv):
        h = np.ones((hsv.shape[0], hsv.shape[1]), dtype=np.uint8) * 10
        s = np.ones((hsv.shape[0], hsv.shape[1]), dtype=np.uint8) * 50
        v = np.ones((hsv.shape[0], hsv.shape[1]), dtype=np.uint8) * 100
        return h, s, v
    cv2_mock.split.side_effect = dummy_split
    
    sys.modules["cv2"] = cv2_mock
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"), threshold=27.0)
    
    assert len(scenes) == 0

def test_detect_video_scenes_with_cuts():
    """Test scene detection when threshold is exceeded."""
    mock_vr = mock.MagicMock()
    mock_vr.__len__.return_value = 100 # ~3 seconds
    mock_vr.get_avg_fps.return_value = 30.0
    mock_vr.__getitem__.return_value.shape = (1080, 1920, 3)
    
    class FakeFramesTensor:
        def detach(self): return self
        def to(self, _): return self
        def numpy(self):
            import numpy as np
            return np.ones((1, 256, 144, 3), dtype=np.uint8) * 128
            
    mock_vr.get_batch.return_value = FakeFramesTensor()
    shorts.VideoReader = mock.MagicMock(return_value=mock_vr)
    
    import numpy as np
    cv2_mock = mock.MagicMock()
    cv2_mock.COLOR_BGR2HSV = 40
    
    call_count = [0]
    def dummy_split(hsv):
        call_count[0] += 1
        val = 10 if call_count[0] % 2 == 0 else 200
        h = np.ones((10, 10), dtype=np.uint8) * val
        return h, h, h
        
    cv2_mock.split.side_effect = dummy_split
    cv2_mock.cvtColor.side_effect = lambda img, mode: img
    sys.modules["cv2"] = cv2_mock
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"), threshold=1.0) 
    
    assert len(scenes) > 1
