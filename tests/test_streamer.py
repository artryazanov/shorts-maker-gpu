from unittest.mock import MagicMock, patch
from unittest import mock

import pytest
import PyNvCodec as nvc
import torch

from shorts_maker.io.streamer import GPUVideoStreamer


def test_streamer_init_success(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    with GPUVideoStreamer(video_path) as streamer:
        assert streamer.video_path == str(video_path)
        assert streamer.target_w == 1920
        assert streamer.target_h == 1080
        assert streamer.start_frame == 0

    with GPUVideoStreamer(video_path, target_width=640, target_height=360) as streamer:
        assert streamer.target_w == 640
        assert streamer.target_h == 360

def test_streamer_init_with_seek(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    with patch.object(nvc.PyFFmpegDemuxer, 'Seek') as mock_seek:
        with GPUVideoStreamer(video_path, seek_time=2.0) as streamer:
            assert streamer.start_frame == 60
        assert mock_seek.called

def test_streamer_init_failure(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    with patch.object(nvc, 'PyNvDecoder', side_effect=Exception("Decode error")):
        with pytest.raises(Exception, match="Decode error"):
            GPUVideoStreamer(video_path)

def test_streamer_stream_batches(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    side_effects = [True] * 10 + [False]
    
    mock_surf = MagicMock()
    mock_surf.Empty.return_value = False
    mock_surf.Height.return_value = 720
    mock_surf.Width.return_value = 1280
    
    with patch.object(nvc.PyFFmpegDemuxer, 'DemuxSinglePacket', side_effect=side_effects):
        mock_decoder = MagicMock()
        mock_decoder.DecodeSurfaceFromPacket.return_value = mock_surf
        with patch.object(nvc, 'PyNvDecoder', return_value=mock_decoder):
            with GPUVideoStreamer(video_path) as streamer:
                batches = list(streamer.stream_batches(batch_size=4, step=1))
                
                assert len(batches) == 3
                assert len(batches[0][0]) == 4
                assert len(batches[0][1]) == 4
                assert len(batches[1][0]) == 4
                assert len(batches[2][0]) == 2

def test_streamer_stream_batches_step(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    side_effects = [True] * 10 + [False]
    mock_surf = MagicMock()
    mock_surf.Empty.return_value = False
    
    with patch.object(nvc.PyFFmpegDemuxer, 'DemuxSinglePacket', side_effect=side_effects):
        mock_decoder = MagicMock()
        mock_decoder.DecodeSurfaceFromPacket.return_value = mock_surf
        with patch.object(nvc, 'PyNvDecoder', return_value=mock_decoder):
            with GPUVideoStreamer(video_path) as streamer:
                batches = list(streamer.stream_batches(batch_size=4, step=2))
                assert len(batches) == 2
                assert len(batches[0][0]) == 4
                assert len(batches[1][0]) == 1

def test_streamer_max_frames(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    side_effects = [True] * 20 + [False]
    mock_surf = MagicMock()
    mock_surf.Empty.return_value = False
    
    with patch.object(nvc.PyFFmpegDemuxer, 'DemuxSinglePacket', side_effect=side_effects):
        mock_decoder = MagicMock()
        mock_decoder.DecodeSurfaceFromPacket.return_value = mock_surf
        with patch.object(nvc, 'PyNvDecoder', return_value=mock_decoder):
            with GPUVideoStreamer(video_path) as streamer:
                batches = list(streamer.stream_batches(batch_size=4, max_frames=5))
                assert len(batches) == 2
                assert len(batches[0][0]) == 4
                # Next batch completes size 4 then breaks
                assert len(batches[1][0]) == 4

def test_streamer_nv12_format(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    with patch("PyNvCodec.PyFFmpegDemuxer.Format", return_value=nvc.PixelFormat.NV12):
        with GPUVideoStreamer(video_path, target_width=1280, target_height=720) as streamer:
            streamer.nv_dmx.DemuxSinglePacket = MagicMock(side_effect=[True, False])
            mock_surf = MagicMock()
            mock_surf.Empty.return_value = False
            streamer.nv_dec.DecodeSurfaceFromPacket.return_value = mock_surf
            
            # Make sure we simulate old VPF throwing TypeError on Execute
            streamer.nv_cvt_yuv.Execute.side_effect = [TypeError, mock.DEFAULT]
            
            batches = list(streamer.stream_batches(batch_size=1))
            assert len(batches) == 1

def test_streamer_seek_with_pts(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    with patch("PyNvCodec.SeekContext"):
        with patch.object(nvc.PyFFmpegDemuxer, 'LastPacketData') as mock_last:
            mock_last.side_effect = lambda pd: setattr(pd, 'pts', 150) # 150 * 0.01 = 1.5s
            
            # Setup packets: First Demux fails to break loop, wait we want to break at 1.5 >= 1.0!
            def demux_side_effect(*args):
                return True
            
            with patch.object(nvc.PyFFmpegDemuxer, 'DemuxSinglePacket', side_effect=[True, True, False]):
                # Target seek 1.0. Time is 1.5, loop breaks immediately
                streamer = GPUVideoStreamer(video_path, seek_time=1.0)
                assert streamer.start_frame == 30 # 30fps

def test_streamer_fallback_make_tensor(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    import PytorchNvCodec as pnvc
    # Temporarily remove make_tensor to test the fallback DptrToTensor
    original_make = getattr(pnvc, "make_tensor", None)
    if original_make:
        delattr(pnvc, "make_tensor")
        
    try:
        with GPUVideoStreamer(video_path) as streamer:
            streamer.nv_dmx.DemuxSinglePacket = MagicMock(side_effect=[True, False])
            mock_surf = MagicMock()
            mock_surf.Empty.return_value = False
            streamer.nv_dec.DecodeSurfaceFromPacket.return_value = mock_surf
            
            batches = list(streamer.stream_batches(batch_size=1))
            assert len(batches) == 1
    finally:
        if original_make:
            setattr(pnvc, "make_tensor", original_make)

def test_streamer_edge_make_tensor_shapes(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    with GPUVideoStreamer(video_path) as streamer:
        streamer.nv_dmx.DemuxSinglePacket = MagicMock(side_effect=[True, True, False])
        mock_surf = MagicMock()
        mock_surf.Empty.return_value = False
        streamer.nv_dec.DecodeSurfaceFromPacket.return_value = mock_surf
        
        # Branch 1: shape (1, 1080, 1920, 3) 
        # Branch 2: shape (3, 1080, 1920) weird permute
        import PytorchNvCodec as pnvc
        
        from tests.mock_gpu import FakeTensor
        pnvc.make_tensor.side_effect = [
            FakeTensor(shape=(1, 1080, 1920, 3)),
            FakeTensor(shape=(3, 1080, 1920)),
        ]
        
        batches = list(streamer.stream_batches(batch_size=2))
        assert batches[0][0].shape[0] == 2 # batched 2 frames

def test_streamer_close_cuda_cache(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    original_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    try:
        with GPUVideoStreamer(video_path):
            pass
        torch.cuda.empty_cache.assert_called_once()
    finally:
        torch.cuda.is_available = original_avail

