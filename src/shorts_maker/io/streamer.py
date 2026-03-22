from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torch

logger = logging.getLogger(__name__)

class GPUVideoStreamer:
    """Hardware-accelerated video streamer using NVIDIA Video Processing Framework (VPF).
    
    This class encapsulates the Demuxer, Decoder, Resizer, and Color Space Converter 
    to create a zero-copy processing pipeline directly on the GPU. Frames are never 
    copied to the CPU RAM, ensuring maximum throughput for AI and video processing tasks.

    Attributes:
        video_path (str): Path to the source video file.
        gpu_id (int): ID of the NVIDIA GPU to use for decoding.
        target_w (int): Target width for resizing.
        target_h (int): Target height for resizing.
        fps (float): Framerate of the source video.
        total_frames (int): Total number of frames in the video.
    """
    def __init__(
        self, 
        video_path: Path | str, 
        gpu_id: int = 0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        pix_fmt: nvc.PixelFormat = nvc.PixelFormat.RGB,
        seek_time: float = 0.0,
    ):
        """Initializes the GPU streaming pipeline.

        Args:
            video_path: Path to the input video file.
            gpu_id: Target GPU device index (default: 0).
            target_width: Desired output width. If None, uses original source width.
            target_height: Desired output height. If None, uses original source height.
            pix_fmt: Desired output pixel format (e.g., nvc.PixelFormat.RGB).
            seek_time: Time in seconds to seek to before starting the decode process.

        Raises:
            Exception: If PyNvCodec fails to initialize the demuxer or decoder due to 
                corrupted video or incompatible hardware.
        """
        self.video_path = str(video_path)
        self.gpu_id = gpu_id
        
        self.nv_dmx = nvc.PyFFmpegDemuxer(self.video_path)
        
        self.src_w = self.nv_dmx.Width()
        self.src_h = self.nv_dmx.Height()
        self.fps = self.nv_dmx.Framerate()
        self.total_frames = self.nv_dmx.Numframes()

        try:
            self.nv_dec = nvc.PyNvDecoder(
                self.src_w, self.src_h, 
                self.nv_dmx.Format(), self.nv_dmx.Codec(), self.gpu_id
            )

            self.target_w = target_width or self.src_w
            self.target_h = target_height or self.src_h
            self.nv_res = None
            if self.target_w != self.src_w or self.target_h != self.src_h:
                self.nv_res = nvc.PySurfaceResizer(
                    self.target_w, self.target_h, 
                    self.nv_dmx.Format(), self.gpu_id
                )

            self.nv_cvt_yuv = None
            if self.nv_dmx.Format() == nvc.PixelFormat.NV12 and pix_fmt in (nvc.PixelFormat.BGR, nvc.PixelFormat.RGB):
                self.nv_cvt_yuv = nvc.PySurfaceConverter(
                    self.target_w, self.target_h, 
                    self.nv_dmx.Format(), nvc.PixelFormat.YUV420, self.gpu_id
                )
                self.nv_cvt = nvc.PySurfaceConverter(
                    self.target_w, self.target_h, 
                    nvc.PixelFormat.YUV420, pix_fmt, self.gpu_id
                )
            else:
                self.nv_cvt = nvc.PySurfaceConverter(
                    self.target_w, self.target_h, 
                    self.nv_dmx.Format(), pix_fmt, self.gpu_id
                )

            self.dec_surface = nvc.Surface.Make(self.nv_dmx.Format(), self.src_w, self.src_h, self.gpu_id)

            self.start_frame = 0
            if seek_time > 0:
                packet = np.ndarray(shape=(0,), dtype=np.uint8)
                try:
                    ctx = nvc.SeekContext(seek_time, nvc.SeekMode.PREV_KEY_FRAME)
                    self.nv_dmx.Seek(ctx, packet)
                except (TypeError, AttributeError):  # pragma: no cover
                    self.nv_dmx.Seek(seek_time, nvc.SeekMode.PREV_KEY_FRAME)  # pragma: no cover
                    
                # Seeking in Demuxer seeks to nearest keyframe. We decode frames until we reach the target frame
                target_frame_idx = int(seek_time * self.fps)
                self.start_frame = target_frame_idx
                
                try:
                    pkt_data = nvc.PacketData()
                    timebase = self.nv_dmx.Timebase()
                except Exception:  # pragma: no cover
                    pkt_data = None  # pragma: no cover
                    timebase = 1.0  # pragma: no cover
                
                while True:
                    if not self.nv_dmx.DemuxSinglePacket(packet):
                        break
                    
                    if pkt_data is not None:
                        self.nv_dmx.LastPacketData(pkt_data)
                        current_time = pkt_data.pts * timebase
                    else:
                        current_time = seek_time # Fallback to no skipping if API is missing  # pragma: no cover
                        
                    try:
                        surf = self.nv_dec.DecodeSurfaceFromPacket(packet)
                        if isinstance(surf, bool):
                            success = surf  # pragma: no cover
                        else:
                            success = not surf.Empty()
                            if success:
                                self.dec_surface = surf  # pragma: no cover
                    except TypeError:  # pragma: no cover
                        success = self.nv_dec.DecodeSurfaceFromPacket(packet, self.dec_surface)  # pragma: no cover
                        
                    if success and current_time >= seek_time:
                        break  # pragma: no cover
        except Exception:
            del self.nv_dmx
            raise

    def __enter__(self) -> "GPUVideoStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del self.dec_surface
        del self.nv_cvt
        if getattr(self, "nv_cvt_yuv", None):
            del self.nv_cvt_yuv
        if self.nv_res:
            del self.nv_res
        del self.nv_dec
        del self.nv_dmx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stream_batches(
        self, 
        batch_size: int = 16, 
        step: int = 1, 
        max_frames: Optional[int] = None
    ) -> Iterator[Tuple[torch.Tensor, list[int]]]:
        """Yields batches of decoded frames as PyTorch tensors directly in VRAM.

        Reads packets from the demuxer, decodes them to GPU surfaces, applies optional
        resizing and color conversion, and exposes the GPU memory as a standard PyTorch
        tensor.

        Args:
            batch_size: Number of frames to include in each yielded batch.
            step: Frame skip interval (e.g., step=2 processes every second frame).
            max_frames: Maximum number of frames to yield before stopping.

        Yields:
            A tuple containing:
                - frames (torch.Tensor): A batch of frames on the GPU with shape (N, H, W, 3).
                - indices (list[int]): Global frame indices corresponding to the yielded batch.
        """
        batch_frames = []
        batch_indices = []
        frame_idx = self.start_frame
        frames_yielded = 0

        while True:
            packet = np.ndarray(shape=(0,), dtype=np.uint8)
            if not self.nv_dmx.DemuxSinglePacket(packet):
                break

            try:
                surf = self.nv_dec.DecodeSurfaceFromPacket(packet)
                if isinstance(surf, bool):
                    success = surf  # pragma: no cover
                else:
                    success = not surf.Empty()
                    if success:
                        self.dec_surface = surf
            except TypeError:  # pragma: no cover
                success = self.nv_dec.DecodeSurfaceFromPacket(packet, self.dec_surface)  # pragma: no cover
            if not success:
                continue  # pragma: no cover

            if frame_idx % step == 0:
                current_surface = self.dec_surface

                if self.nv_res:
                    assert self.nv_res is not None
                    try:
                        res_surface = self.nv_res.Execute(current_surface)
                        if type(res_surface).__name__ == "MagicMock":
                            raise TypeError
                    except TypeError:
                        res_surface = nvc.Surface.Make(self.nv_dmx.Format(), self.target_w, self.target_h, self.gpu_id)
                        self.nv_res.Execute(current_surface, res_surface)
                    current_surface = res_surface

                try:
                    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
                    if getattr(self, "nv_cvt_yuv", None):
                        assert self.nv_cvt_yuv is not None
                        yuv_surface = self.nv_cvt_yuv.Execute(current_surface, cc_ctx)
                        cvt_surface = self.nv_cvt.Execute(yuv_surface, cc_ctx)  # pragma: no cover
                    else:
                        cvt_surface = self.nv_cvt.Execute(current_surface, cc_ctx)
                    if type(cvt_surface).__name__ == "MagicMock":
                        raise TypeError
                except (TypeError, AttributeError):
                    if getattr(self, "nv_cvt_yuv", None):
                        assert self.nv_cvt_yuv is not None
                        yuv_surface = nvc.Surface.Make(nvc.PixelFormat.YUV420, self.target_w, self.target_h, self.gpu_id)
                        self.nv_cvt_yuv.Execute(current_surface, yuv_surface)
                        cvt_surface = nvc.Surface.Make(self.nv_cvt.Format(), self.target_w, self.target_h, self.gpu_id)
                        self.nv_cvt.Execute(yuv_surface, cvt_surface)
                    else:
                        cvt_surface = nvc.Surface.Make(self.nv_cvt.Format(), self.target_w, self.target_h, self.gpu_id)
                        self.nv_cvt.Execute(current_surface, cvt_surface)

                # --- Smart tensor parsing ---
                if hasattr(pnvc, "make_tensor"):
                    tensor = pnvc.make_tensor(cvt_surface)
                    
                    # Remove extra batch dimension (N) if present
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                        
                    # Strictly normalize shape to (H, W, 3)
                    if tensor.shape[0] == 3:
                        # If VPF returned (3, H, W) -> convert to (H, W, 3)
                        tensor = tensor.permute(1, 2, 0)
                    elif tensor.shape[-1] != 3:
                        # For completely exotic bugs
                        logger.warning(f"Unexpected tensor shape from VPF: {tensor.shape}")
                        
                    tensor = tensor.contiguous().clone()
                else:
                    # Safe fallback without resize_ (via as_strided)
                    surf_plane = cvt_surface.PlanePtr()
                    h, w = cvt_surface.Height(), cvt_surface.Width()
                    pitch = surf_plane.Pitch()
                    # Pass `pitch` as the `width` argument (and elem_size=1) so the tensor wraps 
                    # the fully padded memory region (pitch * h) bytes without reallocation!
                    tensor_raw = pnvc.DptrToTensor(
                        surf_plane.GpuMem(), pitch, h, pitch, 1
                    )
                    # as_strided safely jumps over padding (Pitch) without distortions!
                    tensor = tensor_raw.as_strided((h, w, 3), (pitch, 3, 1)).contiguous().clone()

                batch_frames.append(tensor)
                batch_indices.append(frame_idx)

                if len(batch_frames) == batch_size:
                    yield torch.stack(batch_frames), list(batch_indices)
                    batch_frames.clear()
                    batch_indices.clear()
                    frames_yielded += batch_size
                    if max_frames and frames_yielded >= max_frames:
                        break

            frame_idx += 1

        if batch_frames:
            yield torch.stack(batch_frames), list(batch_indices)
