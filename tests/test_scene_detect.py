import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401
from shorts_maker.utils.scenes import detect_video_scenes_gpu  # noqa: E402

@mock.patch("shorts_maker.utils.scenes.GPUVideoStreamer")
@mock.patch("shorts_maker.utils.scenes.nvc.PyFFmpegDemuxer")
def test_detect_video_scenes_empty_video(mock_dmx, mock_streamer):
    """Test scene detection when video has 0 frames."""
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx_instance.Numframes.return_value = 0
    mock_dmx.return_value = mock_dmx_instance
    
    mock_streamer_instance = mock.MagicMock()
    mock_streamer_instance.stream_batches.return_value = []
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"))
    assert len(scenes) == 0

@mock.patch("shorts_maker.utils.scenes.GPUVideoStreamer")
@mock.patch("shorts_maker.utils.scenes.nvc.PyFFmpegDemuxer")
def test_detect_video_scenes_no_cuts(mock_dmx, mock_streamer):
    """Test scene detection when no cuts exist."""
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx_instance.Numframes.return_value = 60
    mock_dmx.return_value = mock_dmx_instance
    
    class FakeFramesTensor:
        def __init__(self, size):
            self.size = size
        def cpu(self): return self
        def numpy(self):
            import numpy as np
            # Return identical gray frames (no scene changes)
            return np.ones((self.size, 256, 144, 3), dtype=np.uint8) * 128
            
    mock_streamer_instance = mock.MagicMock()
    mock_streamer_instance.stream_batches.return_value = [
        (FakeFramesTensor(16), list(range(16))),
        (FakeFramesTensor(16), list(range(16, 32))),
        (FakeFramesTensor(16), list(range(32, 48))),
        (FakeFramesTensor(12), list(range(48, 60))),
    ]
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"), threshold=27.0)
    
    assert len(scenes) == 0

@mock.patch("shorts_maker.utils.scenes.cv2")
@mock.patch("shorts_maker.utils.scenes.GPUVideoStreamer")
@mock.patch("shorts_maker.utils.scenes.nvc.PyFFmpegDemuxer")
def test_detect_video_scenes_with_cuts(mock_dmx, mock_streamer, mock_cv2):
    """Test scene detection when threshold is exceeded."""
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx_instance.Numframes.return_value = 150
    mock_dmx.return_value = mock_dmx_instance
    
    class FakeFramesTensor:
        def __init__(self, size):
            self.size = size
        def cpu(self): return self
        def numpy(self):
            import numpy as np
            return np.ones((self.size, 256, 144, 3), dtype=np.uint8) * 128
            
    mock_streamer_instance = mock.MagicMock()
    batches = []
    for i in range(0, 150, 16):
        end = min(150, i + 16)
        batches.append((FakeFramesTensor(end - i), list(range(i, end))))
    mock_streamer_instance.stream_batches.return_value = batches
    
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context
    
    import numpy as np
    call_count = [0]
    
    def dummy_split(hsv):
        call_count[0] += 1
        val = 10 if (call_count[0] // 50) % 2 == 0 else 200
        h = np.ones((10, 10), dtype=np.uint8) * val
        return h, h, h

    mock_cv2.split.side_effect = dummy_split
    mock_cv2.cvtColor.side_effect = lambda img, mode: img
    mock_cv2.COLOR_BGR2HSV = 40

    scenes = detect_video_scenes_gpu(Path("dummy.mp4"), threshold=1.0) 
    
    assert len(scenes) > 1


@mock.patch("shorts_maker.utils.scenes.GPUVideoStreamer")
@mock.patch("shorts_maker.utils.scenes.nvc.PyFFmpegDemuxer")
def test_detect_video_scenes_small_video(mock_dmx, mock_streamer):
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 200 # < 256
    mock_dmx_instance.Height.return_value = 200
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx_instance.Numframes.return_value = 2
    mock_dmx.return_value = mock_dmx_instance
    
    mock_streamer_instance = mock.MagicMock()
    mock_streamer_instance.stream_batches.return_value = []
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context
    
    scenes = detect_video_scenes_gpu(Path("dummy.mp4"))
    assert len(scenes) == 0

