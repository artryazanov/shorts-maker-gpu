import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401, E402
from shorts_maker.analysis.video import compute_video_action_profile  # noqa: E402

def test_compute_video_action_profile_dmx_error():
    with mock.patch("shorts_maker.analysis.video.nvc.PyFFmpegDemuxer", side_effect=Exception("Bad File")):
        t, s = compute_video_action_profile(Path("dummy.mp4"))
        assert len(t) == 0
        assert len(s) == 0

@mock.patch("shorts_maker.analysis.video.nvc.PyFFmpegDemuxer")
@mock.patch("shorts_maker.analysis.video.GPUVideoStreamer")
def test_compute_video_action_profile_edge_fps(mock_streamer, mock_dmx):
    mock_dmx_inst = mock.MagicMock()
    mock_dmx_inst.Framerate.return_value = 30.0
    mock_dmx_inst.Width.return_value = 1920
    mock_dmx_inst.Height.return_value = 1080
    mock_dmx.return_value = mock_dmx_inst
    
    mock_streamer_inst = mock.MagicMock()
    mock_streamer_inst.total_frames = 100
    # Provide empty stream
    mock_streamer_inst.stream_batches.return_value = []
    
    mock_ctx = mock.MagicMock()
    mock_ctx.__enter__.return_value = mock_streamer_inst
    mock_streamer.return_value = mock_ctx
    
    t, s = compute_video_action_profile(Path("dummy.mp4"), fps=-5)
    assert len(t) == 0
    assert len(s) == 0

@mock.patch("shorts_maker.analysis.video.nvc.PyFFmpegDemuxer")
@mock.patch("shorts_maker.analysis.video.GPUVideoStreamer")
def test_compute_video_action_profile_zero_std(mock_streamer, mock_dmx):
    mock_dmx_inst = mock.MagicMock()
    mock_dmx_inst.Framerate.return_value = 30.0
    mock_dmx_inst.Width.return_value = 1920
    mock_dmx_inst.Height.return_value = 1080
    mock_dmx.return_value = mock_dmx_inst
    
    from tests.mock_gpu import FakeTensor
    mock_streamer_inst = mock.MagicMock()
    mock_streamer_inst.total_frames = 5
    # Provide exactly 1 batch so std() can be 0 (if FakeTensor returns fixed stats)
    tf = FakeTensor(shape=(1, 1080, 1920, 3))
    mock_streamer_inst.stream_batches.return_value = [(tf, [0])]
    
    mock_ctx = mock.MagicMock()
    mock_ctx.__enter__.return_value = mock_streamer_inst
    mock_streamer.return_value = mock_ctx
    
    # Mock std to 0
    with mock.patch("tests.mock_gpu.FakeTensor.std", return_value=0.0):
        t, s = compute_video_action_profile(Path("dummy.mp4"), fps=30)
        assert len(t) == 1
