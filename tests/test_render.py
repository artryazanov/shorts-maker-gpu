import sys
from pathlib import Path
from unittest import mock

import tests.mock_gpu  # noqa: F401
sys.path.append(str(Path(__file__).resolve().parent.parent))
from shorts_maker.io.render import (
    select_background_resolution,
    get_render_params,
    render_video_gpu_isolated,
    render_video_gpu,
)
from shorts_maker.config import ProcessingConfig  # noqa: E402

def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1000) == (900, 1600)
    assert select_background_resolution(1200) == (1080, 1920)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2000) == (1800, 3200)
    assert select_background_resolution(3000) == (2160, 3840)

@mock.patch("shorts_maker.io.render.nvc.PyFFmpegDemuxer")
def test_get_render_params_landscape(mock_dmx):
    config = ProcessingConfig(target_ratio_w=9, target_ratio_h=16)
    
    mock_dmx_instance = mock.MagicMock()
    # Landscape 1920x1080
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx.return_value = mock_dmx_instance
    
    params = get_render_params(Path("dummy.mp4"), 0.0, 15.0, config)
    
    assert params.crop_h == 1080
    assert params.crop_w == 608 # 1080 * 9 / 16
    assert params.bg_width == 720
    assert params.bg_height == 1280
    # Check source
    # w=1920, h=1080. target=9/16. 
    # current_ratio = 1920/1080 = 1.77
    # target_ratio = 9/16 = 0.5625
    # 1.77 > 0.5625 -> crop_w = 1080 * 9 / 16 = 607.5 -> 608.
    # crop_w (608) < crop_h (1080).
    # crop_w / 9 (67) < crop_h / 16 (67.5). So is_vertical_bg is True based on the logic!

@mock.patch("shorts_maker.io.render.nvc.PyFFmpegDemuxer")
def test_get_render_params_portrait(mock_dmx):
    config = ProcessingConfig(target_ratio_w=9, target_ratio_h=16)
    
    mock_dmx_instance = mock.MagicMock()
    # Portrait 1080x1920
    mock_dmx_instance.Width.return_value = 1080
    mock_dmx_instance.Height.return_value = 1920
    mock_dmx.return_value = mock_dmx_instance
    
    params = get_render_params(Path("dummy.mp4"), 0.0, 15.0, config)
    
    # w=1080, h=1920. target=9/16.
    # current_ratio = 1080/1920 = 0.5625
    # target_ratio = 9/16 = 0.5625
    # Crop logic: new_height = 1080 / 9 * 16 = 1920. 
    assert params.crop_h == 1920
    assert params.crop_w == 1080

@mock.patch("shorts_maker.io.render.multiprocessing.get_context")
def test_render_video_gpu_isolated(mock_get_context):
    mock_ctx = mock.MagicMock()
    mock_proc = mock.MagicMock()
    mock_proc.exitcode = 0
    mock_ctx.Process.return_value = mock_proc
    mock_get_context.return_value = mock_ctx
    
    # Just calling it passes 
    render_video_gpu_isolated(None, Path("out.mp4"))
    mock_ctx.Process.assert_called_once()
    mock_proc.start.assert_called_once()
    mock_proc.join.assert_called_once()

@mock.patch("shorts_maker.io.render.GPUVideoStreamer")
@mock.patch("shorts_maker.io.render.nvc.PyFFmpegDemuxer")
@mock.patch("shorts_maker.io.render.subprocess.Popen")
@mock.patch("shorts_maker.io.render.subprocess.run")
def test_render_video_gpu(mock_run, mock_popen, mock_dmx, mock_streamer, tmp_path):
    config = ProcessingConfig()
    
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx.return_value = mock_dmx_instance
    
    mock_streamer_instance = mock.MagicMock()
    mock_streamer_instance.stream_batches.return_value = []
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context

    params = get_render_params(Path("dummy.mp4"), 0.0, 1.0, config)
    
    # Fake process that allows stdin.close() and wait()
    mock_process = mock.MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    
    out_path = tmp_path / "out.mp4"
    render_video_gpu(params, out_path)
    
    # Verify run was called for extracting audio
    mock_run.assert_called_once()
    # Verify Popen was called for ffmpeg encoding
    mock_popen.assert_called_once()
    mock_process.stdin.close.assert_called_once()
    mock_process.wait.assert_called_once()

def test_blur_gpu():
    from shorts_maker.io.render import blur_gpu
    import torch
    
    # Create dummy tensor (H, W, C)
    dummy = torch.ones((10, 10, 3))
    res = blur_gpu(dummy, sigma=2.0)
    assert res is not None
    
    # Create dummy tensor (N, C, H, W)
    dummy_nchw = torch.ones((2, 3, 10, 10))
    res_nchw = blur_gpu(dummy_nchw, sigma=2.0)
    assert res_nchw is not None
    
    # 0 sigma -> returns original
    res_zero = blur_gpu(dummy, sigma=0)
    assert res_zero is dummy

@mock.patch("shorts_maker.io.render.logger")
def test_log_memory_usage(mock_logger):
    from shorts_maker.io.render import log_memory_usage
    log_memory_usage("Test")
    mock_logger.info.assert_called_once()
    assert "Memory:" in mock_logger.info.call_args[0][0]

@mock.patch("shorts_maker.io.render.GPUVideoStreamer")
@mock.patch("shorts_maker.io.render.nvc.PyFFmpegDemuxer")
@mock.patch("shorts_maker.io.render.subprocess.Popen")
@mock.patch("shorts_maker.io.render.subprocess.run")
def test_render_video_gpu_with_frames(mock_run, mock_popen, mock_dmx, mock_streamer, tmp_path):
    from shorts_maker.io.render import get_render_params, render_video_gpu
    
    config = ProcessingConfig()
    
    mock_dmx_instance = mock.MagicMock()
    mock_dmx_instance.Width.return_value = 1920
    mock_dmx_instance.Height.return_value = 1080
    mock_dmx_instance.Framerate.return_value = 30.0
    mock_dmx.return_value = mock_dmx_instance
    
    mock_streamer_instance = mock.MagicMock()
    # Mock stream_batches to yield ONE batch of frames
    fake_frames = tests.mock_gpu.FakeTensor(shape=(4, 1080, 1920, 3))
    mock_streamer_instance.stream_batches.return_value = [(fake_frames, [0, 1, 2, 3], [0.0, 0.033, 0.066, 0.1])]
    mock_streamer_context = mock.MagicMock()
    mock_streamer_context.__enter__.return_value = mock_streamer_instance
    mock_streamer.return_value = mock_streamer_context

    params = get_render_params(Path("dummy.mp4"), 0.0, 1.0, config)
    
    mock_process = mock.MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    
    # Mock select to strictly block / allow write
    with mock.patch("select.select") as mock_select:
        mock_select.return_value = ([], [mock_process.stdin.fileno()], [])
        out_path = tmp_path / "out2.mp4"
        render_video_gpu(params, out_path, save_ffmpeg_logs=True)
    
    # It should have written 4 frames (one batch) to the pipe
    mock_process.stdin.write.assert_called_once()

@mock.patch("shorts_maker.io.render.multiprocessing.get_context")
def test_render_video_gpu_isolated_error(mock_get_context):
    from shorts_maker.io.render import render_video_gpu_isolated
    mock_ctx = mock.MagicMock()
    mock_proc = mock.MagicMock()
    mock_proc.exitcode = -9 # OOM
    mock_ctx.Process.return_value = mock_proc
    mock_get_context.return_value = mock_ctx
    
    # Just calling it passes, log error
    with mock.patch("shorts_maker.io.render.logger") as mock_logger:
        render_video_gpu_isolated(None, Path("out.mp4"))
        assert mock_logger.error.call_count >= 1
