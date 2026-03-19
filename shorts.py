"""Utility for automatically generating short video clips using GPU acceleration.

This module processes gameplay videos and creates resized clips
that fit common short-video aspect ratios. It leverages NVIDIA GPU
(CUDA) for scene detection, audio/video analysis, image filtering,
and video encoding to maximize performance.
"""

from __future__ import annotations
import argparse
import logging
import math
import random
import os
import gc
import multiprocessing
try:
    import resource
except ImportError:
    resource = None
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Iterator
import numpy as np
from dotenv import load_dotenv

import torch
import torchaudio
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
from tqdm import tqdm

# Load environment variables from a .env file if present.
load_dotenv()

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format="%(message)s")


class GPUVideoStreamer:
    """
    Hardware-accelerated video streamer using VPF.
    Encapsulates Demuxer -> Decoder -> Resizer -> Converter.
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
                except (TypeError, AttributeError):
                    self.nv_dmx.Seek(seek_time, nvc.SeekMode.PREV_KEY_FRAME)
                    
                # Seeking in Demuxer seeks to nearest keyframe. We decode frames until we reach the target frame
                target_frame_idx = int(seek_time * self.fps)
                self.start_frame = target_frame_idx
                
                try:
                    pkt_data = nvc.PacketData()
                    timebase = self.nv_dmx.Timebase()
                except Exception:
                    pkt_data = None
                    timebase = 1.0
                
                while True:
                    if not self.nv_dmx.DemuxSinglePacket(packet):
                        break
                    
                    if pkt_data is not None:
                        self.nv_dmx.LastPacketData(pkt_data)
                        current_time = pkt_data.pts * timebase
                    else:
                        current_time = seek_time # Fallback to no skipping if API is missing
                        
                    try:
                        surf = self.nv_dec.DecodeSurfaceFromPacket(packet)
                        if isinstance(surf, bool):
                            success = surf
                        else:
                            success = not surf.Empty()
                            if success:
                                self.dec_surface = surf
                    except TypeError:
                        success = self.nv_dec.DecodeSurfaceFromPacket(packet, self.dec_surface)
                        
                    if success and current_time >= seek_time:
                        break
        except Exception:
            del self.nv_dmx
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

    def stream_batches(self, batch_size: int = 16, step: int = 1, max_frames: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, list[int]]]:
        """
        Gives batches and local indices (from self.start_frame).
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
                    success = surf
                else:
                    success = not surf.Empty()
                    if success:
                        self.dec_surface = surf
            except TypeError:
                success = self.nv_dec.DecodeSurfaceFromPacket(packet, self.dec_surface)
            if not success:
                continue

            if frame_idx % step == 0:
                current_surface = self.dec_surface

                if self.nv_res:
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
                        yuv_surface = self.nv_cvt_yuv.Execute(current_surface, cc_ctx)
                        cvt_surface = self.nv_cvt.Execute(yuv_surface, cc_ctx)
                    else:
                        cvt_surface = self.nv_cvt.Execute(current_surface, cc_ctx)
                    if type(cvt_surface).__name__ == "MagicMock":
                        raise TypeError
                except (TypeError, AttributeError):
                    if getattr(self, "nv_cvt_yuv", None):
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
                        logging.warning(f"Unexpected tensor shape from VPF: {tensor.shape}")
                        
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
                    yield torch.stack(batch_frames), batch_indices
                    batch_frames.clear()
                    batch_indices.clear()
                    frames_yielded += batch_size
                    if max_frames and frames_yielded >= max_frames:
                        break

            frame_idx += 1

        if batch_frames:
            yield torch.stack(batch_frames), batch_indices



def _get_env_int(name: str, default: int) -> int:
    """Read an int environment variable with a default and basic validation."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except Exception:
        logging.warning("Env var %s=%r is not a valid int. Using default %s.", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default and basic validation."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except Exception:
        logging.warning(
            "Env var %s=%r is not a valid float. Using default %s.", name, value, default
        )
        return default


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration values used throughout the processing pipeline."""

    target_ratio_w: int = 1
    target_ratio_h: int = 1
    scene_limit: int = 6
    x_center: float = 0.5
    y_center: float = 0.5
    max_error_depth: int = 3
    min_short_length: int = 15
    max_short_length: int = 179
    max_combined_scene_length: int = 300

    @property
    def middle_short_length(self) -> float:
        """Return the mid point between min and max short lengths."""
        return (self.min_short_length + self.max_short_length) / 2


def log_memory_usage(tag: str = ""):
    """Log current memory usage (RAM and VRAM)."""
    usage_stats = []

    # RAM
    if resource:
        # ru_maxrss is in KB on Linux
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        ram_mb = rusage.ru_maxrss / 1024.0
        usage_stats.append(f"RAM: {ram_mb:.1f} MB")

    # VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        usage_stats.append(f"VRAM Alloc: {allocated:.1f} MB, Res: {reserved:.1f} MB")

    logging.info(f"[{tag}] Memory: {', '.join(usage_stats)}")


@dataclass
class RenderParams:
    """Parameters required to render the final clip."""
    source_path: Path
    start_time: float
    duration: float
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    bg_width: int
    bg_height: int
    is_vertical_bg: bool  # True if 9:16 background, False if 1:1 background (resizing logic)


class _SecondsTime:
    """Lightweight stand-in for scene time objects using seconds."""

    def __init__(self, seconds: float):
        self._seconds = float(seconds)

    def get_seconds(self) -> float:
        return self._seconds

    def get_timecode(self) -> str:
        return f"{self._seconds:.2f}"

    def get_frames(self) -> int:
        return int(self._seconds * 30)


def detect_video_scenes_gpu(video_path: Path, threshold: float = 27.0) -> List[Tuple[_SecondsTime, _SecondsTime]]:
    """Detect scenes matching PySceneDetect ContentDetector, but with GPU-assisted I/O.

    This implementation replicates scenedetect.detectors.ContentDetector (v0.6.7)
    semantics to produce identical scene cuts:
      - Frames are downscaled to an effective width of ~256 px (like SceneManager.auto_downscale).
      - Frame score is the mean absolute difference between adjacent frames in HSV space
        (per-channel: hue, saturation, value), averaged with equal weights.
      - Cuts are produced via the same FlashFilter MERGE policy with min_scene_len=15 frames.
      - Scene list is generated exactly like SceneManager.get_scene_list(start_in_scene=False):
        if no cuts are found, returns an empty list.

    GPU usage: frames are decoded/resized with decord (GPU if available). HSV conversion uses
    OpenCV on CPU to match ContentDetector exactly. The difference/thresholding logic follows
    the original algorithm.
    """
    import cv2

    # 1) Determine original size, compute SceneDetect-like downscale factor.
    dmx = nvc.PyFFmpegDemuxer(str(video_path))
    w0 = dmx.Width()
    h0 = dmx.Height()
    fps = dmx.Framerate()
    frame_count = dmx.Numframes()
    del dmx

    # SceneManager.DEFAULT_MIN_WIDTH = 256
    TARGET_MIN_WIDTH = 256
    if w0 < TARGET_MIN_WIDTH:
        downscale = 1.0
    else:
        downscale = w0 / float(TARGET_MIN_WIDTH)

    w_eff = int(w0 / downscale)
    h_eff = int(h0 / downscale)
    w_eff = max(1, w_eff)
    h_eff = max(1, h_eff)

    if frame_count == 0 or fps <= 0.0:
        return []

    # 3) FlashFilter (MERGE) identical logic to scenedetect.scene_detector.FlashFilter
    class _FlashFilterMerge:
        def __init__(self, length: int):
            self._filter_length = int(length)
            self._last_above: Optional[int] = None
            self._merge_enabled: bool = False
            self._merge_triggered: bool = False
            self._merge_start: Optional[int] = None

        @property
        def max_behind(self) -> int:
            return self._filter_length

        def filter(self, frame_num: int, above_threshold: bool) -> List[int]:
            if not (self._filter_length > 0):
                return [frame_num] if above_threshold else []
            if self._last_above is None:
                self._last_above = frame_num
            # MERGE path
            return self._filter_merge(frame_num, above_threshold)

        def _filter_merge(self, frame_num: int, above_threshold: bool) -> List[int]:
            min_length_met = (frame_num - self._last_above) >= self._filter_length
            if above_threshold:
                self._last_above = frame_num
            if self._merge_triggered:
                num_merged_frames = self._last_above - self._merge_start
                if min_length_met and (not above_threshold) and (num_merged_frames >= self._filter_length):
                    self._merge_triggered = False
                    return [self._last_above]
                return []
            if not above_threshold:
                return []
            if min_length_met:
                self._merge_enabled = True
                return [frame_num]
            if self._merge_enabled:
                self._merge_triggered = True
                self._merge_start = frame_num
            return []

    min_scene_len = 15  # ContentDetector default
    flash_filter = _FlashFilterMerge(length=min_scene_len)

    # 4) Iterate frames, compute HSV components & frame score like ContentDetector
    #    Score normalization: divide by sum(abs(weights)) = 3.
    batch_size = 16
    total_batches = (frame_count + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Detect scenes", unit="batch")

    last_hsv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    cut_indices: List[int] = []

    with GPUVideoStreamer(
        video_path, 
        target_width=w_eff, 
        target_height=h_eff, 
        pix_fmt=nvc.PixelFormat.BGR
    ) as streamer:
        for frames_bgr, batch_indices in streamer.stream_batches(batch_size=batch_size):
            frames_cpu = frames_bgr.cpu().numpy()

            # Process each frame sequentially to exactly match CPU semantics
            for j, bgr in enumerate(frames_cpu):
                frame_num = batch_indices[j]
                bgr = np.ascontiguousarray(bgr)

                # OpenCV HSV conversion (exact semantics/hue range)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hue, sat, val = cv2.split(hsv)

                if last_hsv is None:
                    last_hsv = (hue, sat, val)
                    # First score is 0.0 by design
                    above = False
                    # Prime flash filter state
                    flash_filter.filter(frame_num, above_threshold=above)
                    continue

                hue_prev, sat_prev, val_prev = last_hsv
                # Mean pixel distance per channel (match _mean_pixel_distance)
                # cast to int32 to avoid uint8 underflow
                dh = np.abs(hue.astype(np.int32) - hue_prev.astype(np.int32)).sum() / float(hue.size)
                ds = np.abs(sat.astype(np.int32) - sat_prev.astype(np.int32)).sum() / float(sat.size)
                dv = np.abs(val.astype(np.int32) - val_prev.astype(np.int32)).sum() / float(val.size)
                frame_score = (dh + ds + dv) / 3.0

                # Record and advance last_hsv
                last_hsv = (hue, sat, val)

                # Compare against threshold exactly like ContentDetector
                above = frame_score >= threshold
                emitted = flash_filter.filter(frame_num=frame_num, above_threshold=above)
                if emitted:
                    cut_indices.extend(emitted)

            pbar.update(1)

    pbar.close()

    # Build scenes like get_scenes_from_cuts, but align to detect_video_scenes default (start_in_scene=False)
    if not cut_indices:
        return []

    cut_indices = sorted(set(cut_indices))
    scenes: List[Tuple[_SecondsTime, _SecondsTime]] = []
    last_cut = 0
    for cut in cut_indices:
        start_time = last_cut / fps
        end_time = cut / fps
        scenes.append((_SecondsTime(start_time), _SecondsTime(end_time)))
        last_cut = cut
    # Last scene from last cut to end_pos (= frame_count, exclusive)
    scenes.append((_SecondsTime(last_cut / fps), _SecondsTime(frame_count / fps)))

    return scenes


def blur_gpu(image_tensor: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Return a blurred version of an image using native PyTorch separable convolutions.
    Accepts both (H, W, C) and (N, C, H, W) formats.
    """
    if sigma <= 0:
        return image_tensor
        
    # Determine format and convert to (N, C, H, W)
    is_hwc = image_tensor.dim() == 3
    if is_hwc:
        x = image_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
    else:
        x = image_tensor.float()

    channels = x.shape[1]
    
    # Kernel radius (typically 3 * sigma)
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    
    # Create 1D Gaussian kernel
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Reshape for depthwise convolution (per-channel)
    kernel_y = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    kernel_x = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    
    # Add padding (reflect mode to avoid black edges)
    x = torch.nn.functional.pad(x, (radius, radius, radius, radius), mode='reflect')
    
    # Apply separable convolutions (Y first, then X)
    x = torch.nn.functional.conv2d(x, kernel_y, groups=channels)
    x = torch.nn.functional.conv2d(x, kernel_x, groups=channels)
    
    # Revert to original format
    if is_hwc:
        x = x.squeeze(0).permute(1, 2, 0)
        
    return x.to(image_tensor.dtype)


# --- Audio-based action scoring (GPU) -------------------------------------------

@torch.no_grad()
def compute_audio_action_profile(
    video_path: Path,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute audio-based "action score" on GPU with memory-efficient batching.

    Returns:
      times  - array of times (seconds) for each feature frame
      score  - combined action score (loudness + spectral "roughness")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        info = torchaudio.info(str(video_path))
        sample_rate = info.sample_rate
        total_samples = info.num_frames
    except Exception:
        logging.error(f"Failed to load audio from {video_path}")
        return np.array([]), np.array([])
    
    rms_values = []
    flux_values = []
    
    window = torch.hann_window(2048).to(device)
    last_mag_col = torch.zeros(2048 // 2 + 1, device=device)
    
    # Process in chunks of 2 minutes to save RAM
    # Make sure chunk_frames is a multiple of hop_length
    chunk_frames = (sample_rate * 120 // hop_length) * hop_length
    # Overlap allows STFT and RMS to be seamless across boundaries
    overlap_frames = frame_length
    
    current_frame = 0
    pbar = tqdm(total=total_samples if total_samples > 0 else 1, desc="Audio Profile", unit="samples")
    
    while current_frame < total_samples or total_samples <= 0:
        read_count = chunk_frames + (overlap_frames if current_frame > 0 else 0)
        read_start = max(0, current_frame - overlap_frames)
        
        try:
            waveform, sr = torchaudio.load(
                str(video_path),
                frame_offset=read_start,
                num_frames=read_count,
                normalize=True
            )
        except Exception:
            logging.error(f"Error reading audio chunk at {read_start}")
            break
            
        if waveform.shape[1] == 0:
            break
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        y_cpu = waveform.squeeze(0)
        actual_length = y_cpu.shape[0]
        
        if actual_length < frame_length:
            y_cpu = torch.nn.functional.pad(y_cpu, (0, frame_length - actual_length))
            
        chunk_tensor = y_cpu.to(device)
        
        # --- RMS ---
        windows = chunk_tensor.unfold(0, frame_length, hop_length)
        rms_chunk = torch.sqrt(torch.mean(windows**2, dim=1))
        rms_values.append(rms_chunk)
        
        # --- STFT ---
        # reflect pad
        y_padded = torch.nn.functional.pad(chunk_tensor.unsqueeze(0), (2048 // 2, 2048 // 2), mode='reflect').squeeze(0)
        stft_chunk = torch.stft(
            y_padded, 
            n_fft=2048, 
            hop_length=hop_length, 
            window=window, 
            center=False, 
            return_complex=True
        )
        mag_chunk = torch.abs(stft_chunk)
        
        combined = torch.cat([last_mag_col.unsqueeze(1), mag_chunk], dim=1)
        diff = combined[:, 1:] - combined[:, :-1]
        flux_chunk = torch.sqrt(torch.sum(diff**2, dim=0))
        flux_values.append(flux_chunk)
        
        last_mag_col = mag_chunk[:, -1]
        
        del chunk_tensor, windows, y_padded, stft_chunk, mag_chunk, combined, diff
        
        actual_forward = actual_length - (overlap_frames if current_frame > 0 else 0)
        if actual_forward <= 0:
            break
            
        current_frame += actual_forward
        pbar.update(actual_forward)
        
        # EOF detection
        if actual_length < read_count:
            break

    pbar.close()
    
    rms = torch.cat(rms_values) if rms_values else torch.tensor([], device=device)
    spectral_flux = torch.cat(flux_values) if flux_values else torch.tensor([], device=device)
    
    # --- Post Processing ---
    min_len = min(rms.shape[0], spectral_flux.shape[0])
    rms = rms[:min_len]
    spectral_flux = spectral_flux[:min_len]

    rms_mean = rms.mean() if rms.numel() > 0 else torch.tensor(0.0, device=device)
    rms_std = (rms.std() + 1e-8) if rms.numel() > 0 else torch.tensor(1.0, device=device)
    rms_norm = (rms - rms_mean) / rms_std if rms.numel() > 0 else rms

    flux_mean = spectral_flux.mean() if spectral_flux.numel() > 0 else torch.tensor(0.0, device=device)
    flux_std = (spectral_flux.std() + 1e-8) if spectral_flux.numel() > 0 else torch.tensor(1.0, device=device)
    flux_norm = (spectral_flux - flux_mean) / flux_std if spectral_flux.numel() > 0 else spectral_flux

    def smooth_gpu(x: torch.Tensor, win: int = 21) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if win > x.shape[0]:
            win = x.shape[0]
        if win % 2 == 0:
            win += 1
        padding = win // 2
        kernel = torch.ones(win, device=device) / win
        x_reshaped = x.view(1, 1, -1)
        kernel_reshaped = kernel.view(1, 1, -1)
        out = torch.nn.functional.conv1d(x_reshaped, kernel_reshaped, padding=padding)
        return out.view(-1)

    rms_smooth = smooth_gpu(rms_norm, win=21)
    flux_smooth = smooth_gpu(flux_norm, win=21)

    score = 0.6 * rms_smooth + 0.4 * flux_smooth if rms_smooth.numel() > 0 and flux_smooth.numel() > 0 else (
        rms_smooth if flux_smooth.numel() == 0 else flux_smooth
    )

    num_frames_out = score.shape[0]
    times = torch.arange(num_frames_out, device=device) * hop_length / sample_rate if num_frames_out > 0 else torch.tensor([], device=device)

    return times.cpu().numpy(), score.cpu().numpy()


@torch.no_grad()
def compute_video_action_profile(
    video_path: Path,
    fps: int = 6,
    downscale_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute video-based "action score" on GPU.

    Uses Decord to read frames directly to GPU memory and computes
    mean absolute pixel difference.

    Robust to DECORD EOF issues: wraps get_batch with retries/chunking and
    allows configuring DECORD_EOF_RETRY_MAX via environment.
    """

    # 1) Get metadata and calculate dimensions
    try:
        dmx = nvc.PyFFmpegDemuxer(str(video_path))
        orig_fps = float(dmx.Framerate())
        w_new = max(1, dmx.Width() // downscale_factor)
        h_new = max(1, dmx.Height() // downscale_factor)
        del dmx
    except Exception:
        logging.warning("Failed to load video for action profile.", exc_info=True)
        return np.array([]), np.array([])

    eff_fps = min(float(fps), orig_fps)
    if eff_fps <= 0:
        eff_fps = max(1.0, float(fps))

    # Calculate step for subsampling
    step = max(1, int(orig_fps / eff_fps))

    motions = []
    times = []
    prev_batch_last = None

    with GPUVideoStreamer(video_path, target_width=w_new, target_height=h_new) as streamer:
        total_batches = int(np.ceil(streamer.total_frames / (step * 16)))
        pbar = tqdm(total=total_batches, desc="Video Action Profile", unit="batch")
        
        # GPUVideoStreamer natively handles iterating to the end without hanging
        # and outputs only the batches representing the requested `step`
        for frames_subset, global_indices in streamer.stream_batches(batch_size=16, step=step):
            frames_subset = frames_subset.float()
            
            # Grayscale conversion on GPU
            gray = (frames_subset[..., 0] * 0.299 +
                    frames_subset[..., 1] * 0.587 +
                    frames_subset[..., 2] * 0.114)

            # Diff computation
            if prev_batch_last is not None:
                combined = torch.cat([prev_batch_last.unsqueeze(0), gray])
                diffs = torch.abs(combined[1:] - combined[:-1])
            else:
                combined = torch.cat([gray[0:1], gray])
                diffs = torch.abs(combined[1:] - combined[:-1])
                diffs[0] = 0.0

            # Mean diff per frame
            batch_motions = diffs.mean(dim=(1, 2))
            motions.append(batch_motions)

            # Timestamps
            batch_times = torch.tensor(global_indices, device=gray.device).float() / orig_fps
            times.append(batch_times)

            # Update last processed frame for next continuity
            prev_batch_last = gray[-1]

            del frames_subset, gray, diffs
            
            pbar.update(1)
        
        pbar.close()

    if len(motions) == 0:
        return np.array([]), np.array([])

    motions = torch.cat(motions)
    times = torch.cat(times)

    # Normalize and smooth (similar to audio)
    if motions.numel() == 0:
        return np.array([]), np.array([])
    if motions.std() == 0:
        motions_norm = motions
    else:
        motions_norm = (motions - motions.mean()) / (motions.std() + 1e-8)

    # Smooth
    def smooth_gpu(x, win):
        if win > x.shape[0]:
            win = x.shape[0]
        if win < 2:
            return x
        kernel = torch.ones(win, device=x.device) / win
        x_reshaped = x.view(1, 1, -1)
        kernel_reshaped = kernel.view(1, 1, -1)
        out = torch.nn.functional.conv1d(x_reshaped, kernel_reshaped, padding=win//2)
        return out.view(-1)[:x.shape[0]]

    score = smooth_gpu(motions_norm, win=int(eff_fps))

    return times.cpu().numpy(), score.cpu().numpy()


def scene_action_score(
    scene: Tuple,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Return total (summed) action score within the scene."""

    start_sec = scene[0].get_seconds()
    end_sec = scene[1].get_seconds()

    if end_sec <= start_sec:
        return 0.0

    def _segment_sum(times: np.ndarray, score: np.ndarray) -> float:
        if times.size == 0 or score.size == 0:
            return 0.0
        mask = (times >= start_sec) & (times < end_sec)
        if not np.any(mask):
            return 0.0
        return float(score[mask].sum())

    audio_val = _segment_sum(audio_times, audio_score)

    if video_times is None or video_score is None:
        return audio_val

    video_val = _segment_sum(video_times, video_score)

    return w_audio * audio_val + w_video * video_val


def _best_window_single(
    scene: Tuple,
    window_length: float,
    times: np.ndarray,
    score: np.ndarray,
) -> float:
    """Helper to find best window on a single profile."""

    start_sec = float(scene[0].get_seconds())
    end_sec = float(scene[1].get_seconds())

    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return start_sec

    max_allowed_start = end_sec - float(window_length)
    if max_allowed_start <= start_sec:
        return max(start_sec, min(start_sec, end_sec - float(window_length)))

    mask = (times >= start_sec) & (times <= end_sec)
    if not np.any(mask):
        return start_sec

    t_seg = times[mask]
    s_seg = score[mask]

    if len(t_seg) < 2:
        return start_sec

    dt = float(np.median(np.diff(t_seg)))
    if not math.isfinite(dt) or dt <= 0:
        return start_sec

    n_win = int(max(1, round(float(window_length) / dt)))
    if len(s_seg) < n_win:
        return start_sec

    csum = np.cumsum(np.concatenate(([0.0], s_seg)))
    window_sums = csum[n_win:] - csum[:-n_win]
    best_idx = int(np.argmax(window_sums))

    best_start_time = float(t_seg[best_idx])
    best_start_time = max(start_sec, min(best_start_time, max_allowed_start))

    return best_start_time


def best_action_window_start(
    scene: Tuple,
    window_length: float,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Find the start of the window inside the scene maximizing combined action."""

    if (
        video_times is None
        or video_score is None
        or len(video_times) == 0
        or len(video_score) == 0
    ):
        return _best_window_single(scene, window_length, audio_times, audio_score)

    start_sec = float(scene[0].get_seconds())
    end_sec = float(scene[1].get_seconds())

    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return start_sec

    a_mask = (audio_times >= start_sec) & (audio_times <= end_sec)
    if not np.any(a_mask):
        return _best_window_single(scene, window_length, video_times, video_score)

    t_a_seg = audio_times[a_mask]
    s_a_seg = audio_score[a_mask]

    if len(t_a_seg) < 2:
        return _best_window_single(scene, window_length, video_times, video_score)

    if len(video_times) > 1:
        order = np.argsort(video_times)
        v_interp = np.interp(t_a_seg, video_times[order], video_score[order])
    else:
        v_interp = np.full_like(t_a_seg, float(video_score[0]), dtype=float)

    combined_seg = w_audio * s_a_seg + w_video * v_interp

    dt = float(np.median(np.diff(t_a_seg)))
    if not math.isfinite(dt) or dt <= 0:
        return _best_window_single(scene, window_length, audio_times, audio_score)

    max_allowed_start = end_sec - float(window_length)
    if max_allowed_start <= start_sec:
        return max(start_sec, min(start_sec, end_sec - float(window_length)))

    n_win = int(max(1, round(float(window_length) / dt)))
    if len(combined_seg) < n_win:
        return _best_window_single(scene, window_length, audio_times, audio_score)

    csum = np.cumsum(np.concatenate(([0.0], combined_seg)))
    window_sums = csum[n_win:] - csum[:-n_win]
    best_idx = int(np.argmax(window_sums))

    best_start_time = float(t_a_seg[best_idx])
    best_start_time = max(start_sec, min(best_start_time, max_allowed_start))

    return best_start_time


def select_background_resolution(width: int) -> Tuple[int, int]:
    """Choose an output resolution based on the clip width."""
    if width < 840:
        return 720, 1280
    if width < 1020:
        return 900, 1600
    if width < 1320:
        return 1080, 1920
    if width < 1680:
        return 1440, 2560
    if width < 2040:
        return 1800, 3200
    return 2160, 3840


def get_render_params(
    video_path: Path,
    start_point: float,
    final_clip_length: float,
    config: ProcessingConfig,
) -> RenderParams:
    """Calculate all parameters needed for rendering the final clip."""

    # Use PyFFmpegDemuxer to get dimensions quickly
    dmx = nvc.PyFFmpegDemuxer(str(video_path))
    w = dmx.Width()
    h = dmx.Height()

    # Calculate crop parameters (same logic as before: crop to target ratio)
    current_ratio = w / h
    target_ratio = config.target_ratio_w / config.target_ratio_h

    if current_ratio > target_ratio:
        # Too wide, crop width
        new_width = round(h * config.target_ratio_w / config.target_ratio_h)
        crop_w = new_width
        crop_h = h
        crop_x = int(w * config.x_center - crop_w / 2)
        crop_y = int(h * config.y_center - crop_h / 2)
    else:
        # Too tall, crop height
        new_height = round(w / config.target_ratio_w * config.target_ratio_h)
        crop_w = w
        crop_h = new_height
        crop_x = int(w * config.x_center - crop_w / 2)
        crop_y = int(h * config.y_center - crop_h / 2)

    # Clamp crop coordinates
    crop_x = max(0, min(w - crop_w, crop_x))
    crop_y = max(0, min(h - crop_h, crop_y))

    # Calculate background/output resolution
    bg_w, bg_h = select_background_resolution(crop_w)

    # Logic from get_final_clip to determine layout
    is_vertical_bg = False

    if crop_w >= crop_h:
        # Landscape/Squareish
        # "background_clip = background_clip.resized(width=720, height=720)"
        # "result_clip = result_clip.resized(width=bg_w)" -> final output is bg_w x bg_w
        # This implies we want a square output if the main clip is landscape/square
        bg_h = bg_w # Force square output
        is_vertical_bg = False
    elif crop_w / 9 < crop_h / 16:
         # Very tall portrait
         is_vertical_bg = True
    else:
        # Default fallback
        pass

    return RenderParams(
        source_path=video_path,
        start_time=start_point,
        duration=final_clip_length,
        output_width=bg_w,
        output_height=bg_h,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        bg_width=bg_w,
        bg_height=bg_h,
        is_vertical_bg=is_vertical_bg
    )


def render_video_gpu(
        params: RenderParams,
        output_path: Path,
        max_error_depth: int = 3,
) -> None:
    """Render the clip using GPU compositing and FFMPEG NVENC (Optimized)."""

    logging.info(f"Rendering GPU: {output_path.name}")

    # 1. Fast Extract Audio
    src_path = Path(params.source_path)
    temp_audio = output_path.with_suffix(".aac")

    cmd_audio = [
        "/usr/bin/ffmpeg", "-y",
        "-ss", f"{params.start_time:.3f}",
        "-i", str(params.source_path),
        "-t", f"{params.duration:.3f}",
        "-q:a", "0",
        "-map", "a?",
        str(temp_audio)
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # 2. Setup FFMPEG process
    fps = 30.0
    try:
        dmx = nvc.PyFFmpegDemuxer(str(params.source_path))
        src_fps = float(dmx.Framerate())
        fps = min(src_fps, 60.0)
        del dmx
    except Exception:
        fps = 30.0

    cmd_ffmpeg = [
        "/usr/bin/ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{params.output_width}x{params.output_height}",
        "-pix_fmt", "rgb24",
        "-r", f"{fps}",
        "-i", "-",
    ]

    has_valid_audio = temp_audio.exists() and temp_audio.stat().st_size > 0

    if has_valid_audio:
        cmd_ffmpeg.extend(["-i", str(temp_audio)])

    cmd_ffmpeg.extend([
        "-c:v", "hevc_nvenc",  # Use hardware encoder
        "-preset", "slow",     # 'slow' is compatible with old and new ffmpeg NVENC
        "-rc", "vbr",
        "-cq", "23",  # Slightly increased CQ to reduce bitrate spikes
        "-maxrate", "80M",  # Cap bitrate to prevent buffer bloat
        "-bufsize", "100M",
        "-pix_fmt", "yuv420p",
        "-g", f"{int(fps * 2)}",
        "-bf", "2",
    ])

    if has_valid_audio:
        cmd_ffmpeg.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd_ffmpeg.extend(["-shortest", str(output_path)])

    # redirect stderr to a file to prevent buffer deadlock
    log_path = output_path.with_suffix(".ffmpeg.log")
    ffmpeg_log = open(log_path, "w")

    try:
        process = subprocess.Popen(
            cmd_ffmpeg,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log
        )
    except Exception:
        ffmpeg_log.close()
        raise

    # 3. GPU Rendering Loop
    # Use torch.no_grad() to prevent graph building overhead
    with torch.no_grad():
        try:
            fg_w = params.output_width
            fg_h = int(params.crop_h * (params.output_width / params.crop_w))

            dmx = nvc.PyFFmpegDemuxer(str(params.source_path))
            src_h = dmx.Height()
            src_w = dmx.Width()
            src_fps_val = float(dmx.Framerate())
            del dmx

            if params.is_vertical_bg:
                bg_ratio_w, bg_ratio_h = 9, 16
                if (src_w / src_h) > (bg_ratio_w / bg_ratio_h):
                    bg_crop_h, bg_crop_w = src_h, int(src_h * bg_ratio_w / bg_ratio_h)
                else:
                    bg_crop_w, bg_crop_h = src_w, int(src_w * bg_ratio_h / bg_ratio_w)
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(src_h * 0.5 - bg_crop_h / 2)
            else:
                bg_dim = min(src_w, src_h)
                bg_crop_w, bg_crop_h = bg_dim, bg_dim
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(src_h * 0.5 - bg_crop_h / 2)

            total_frames = int(params.duration * fps)

            BATCH_SIZE = 4
            total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE

            log_memory_usage("Render Start")

            with tqdm(total=total_batches, desc="Video render", unit="batch") as pbar_render, \
                 GPUVideoStreamer(params.source_path, seek_time=params.start_time) as streamer:
                 
                batch_count = 0
                for frames, _ in streamer.stream_batches(batch_size=BATCH_SIZE, max_frames=total_frames):
                    if batch_count % 50 == 0:
                        logging.info(f"Rendering batch {batch_count}/{total_batches}")
                    batch_count += 1

                    if process.poll() is not None:
                        logging.error("FFMPEG died")
                        break

                    # 1. Background Processing
                    bg_frames = frames[:, bg_crop_y:bg_crop_y + bg_crop_h, bg_crop_x:bg_crop_x + bg_crop_w, :]
                    bg_frames = bg_frames.permute(0, 3, 1, 2).float()  # to NCHW

                    # Resize for blur (low res)
                    blur_w, blur_h = 720, (1280 if params.is_vertical_bg else 720)
                    bg_small = torch.nn.functional.interpolate(
                        bg_frames, size=(blur_h, blur_w), mode='bilinear', align_corners=False
                    )

                    # Blur via Native PyTorch (bg_small is already NCHW)
                    blurred_bg = blur_gpu(bg_small, sigma=16.0)
                    final_bg = torch.nn.functional.interpolate(
                        blurred_bg, size=(params.output_height, params.output_width), mode='bilinear',
                        align_corners=False
                    )

                    # 2. Foreground Processing
                    fg_frames = frames[
                        :, params.crop_y:params.crop_y + params.crop_h, params.crop_x:params.crop_x + params.crop_w, :]
                    fg_frames = fg_frames.permute(0, 3, 1, 2).float()
                    final_fg = torch.nn.functional.interpolate(
                        fg_frames, size=(fg_h, fg_w), mode='bilinear', align_corners=False
                    )

                    # 3. Composite (Overlay)
                    y_off, x_off = (params.output_height - fg_h) // 2, (params.output_width - fg_w) // 2
                    y1, y2 = max(0, y_off), min(params.output_height, y_off + fg_h)
                    x1, x2 = max(0, x_off), min(params.output_width, x_off + fg_w)
                    sy1, sx1 = max(0, -y_off), max(0, -x_off)

                    # Direct tensor insertion
                    if (y2 > y1) and (x2 > x1):
                        final_bg[:, :, y1:y2, x1:x2] = final_fg[:, :, sy1:(sy1 + (y2 - y1)), sx1:(sx1 + (x2 - x1))]

                    # 4. Write to Pipe
                    # Convert to byte and move to CPU
                    # .detach() ensures no grad tracking (redundant with torch.no_grad but safe)
                    out_tensor = final_bg.permute(0, 2, 3, 1).contiguous().byte()
                    out_bytes = out_tensor.cpu().numpy().tobytes()

                    try:
                        import select
                        _, writable, _ = select.select([], [process.stdin.fileno()], [], 10.0)
                        if not writable:
                            logging.error("FFMPEG stdin write blocked (timeout)")
                            break
                        process.stdin.write(out_bytes)
                    except BrokenPipeError:
                        break

                    # 5. Explicit Cleanup (Critical for Loop)
                    del frames, bg_frames, bg_small, blurred_bg, final_bg, fg_frames, final_fg, out_tensor, out_bytes

                    # Periodic GC - keep it, but less frequent is fine
                    if batch_count > 0 and batch_count % 100 == 0:
                        gc.collect()

                    pbar_render.update(1)

        except Exception as e:
            logging.error(f"Error during GPU render: {e}", exc_info=True)
        finally:
            # Clean up processes and memory
            if process:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.wait()

            if 'ffmpeg_log' in locals() and ffmpeg_log:
                ffmpeg_log.close()

            if temp_audio.exists():
                temp_audio.unlink()


            # Final memory sweep
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()


def render_video_gpu_isolated(*args, **kwargs) -> None:
    """Runs render_video_gpu in a separate process to ensure memory cleanup."""
    # Use 'spawn' to ensure a fresh process space
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(
        target=render_video_gpu,
        args=args,
        kwargs=kwargs
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        logging.error("Render process failed with exit code %s", p.exitcode)
        # If exit code is 137 (128+9) or -9, it was likely OOM killed.
        if p.exitcode == -9 or p.exitcode == 137:
             logging.error("Render process was likely OOM killed.")


def combine_scenes(scene_list: Sequence[Tuple], config: ProcessingConfig) -> List[List]:
    """Combine adjacent scenes while preserving content."""

    if not scene_list:
        return []

    def is_small(scene) -> bool:
        return (scene[1].get_seconds() - scene[0].get_seconds()) < config.min_short_length

    n = len(scene_list)
    out: List[List] = []

    # Initialize first run
    run_start_idx = 0
    run_type_small = is_small(scene_list[0])
    run_start_time = scene_list[0][0]
    run_end_time = scene_list[0][1]

    for i in range(1, n):
        current_small = is_small(scene_list[i])
        if current_small == run_type_small:
            # Same-type run continues; extend end.
            run_end_time = scene_list[i][1]

            if run_type_small:
                run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
                if run_duration > config.max_combined_scene_length:
                    prev_end_time = scene_list[i - 1][1]
                    out.append([run_start_time, prev_end_time])
                    run_start_idx = i
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                elif run_duration == config.max_combined_scene_length:
                    is_last_scene = (i == n - 1)
                    if is_last_scene:
                        prev_end_time = scene_list[i - 1][1]
                        out.append([run_start_time, prev_end_time])
                        run_start_idx = i
                        run_start_time = scene_list[i][0]
                        run_end_time = scene_list[i][1]
                    else:
                        out.append([run_start_time, run_end_time])
                        run_start_idx = i + 1
                        run_start_time = scene_list[i][1]
                        run_end_time = scene_list[i][1]
        else:
            run_end_idx = i - 1
            run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
            is_boundary = (run_start_idx == 0) or (run_end_idx == n - 1)
            threshold = config.middle_short_length if is_boundary else config.min_short_length

            if run_duration >= threshold:
                out.append([run_start_time, run_end_time])
                run_start_idx = i
                run_type_small = current_small
                run_start_time = scene_list[i][0]
                run_end_time = scene_list[i][1]
            else:
                if is_boundary and run_start_idx == 0:
                    run_start_idx = i
                    run_type_small = current_small
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                else:
                    run_type_small = current_small
                    run_end_time = scene_list[i][1]

    final_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
    is_boundary = True
    threshold = config.middle_short_length if is_boundary else config.min_short_length
    if final_duration >= threshold:
        out.append([run_start_time, run_end_time])

    return out


def split_overlong_scenes(combined_scene_list: List[List], config: ProcessingConfig) -> List[List]:
    """Split scenes longer than 4 * max_short_length into n equal parts."""
    result: List[List] = []
    threshold = 4 * config.max_short_length
    for scene in combined_scene_list:
        start_s = scene[0].get_seconds()
        end_s = scene[1].get_seconds()
        duration = end_s - start_s

        if duration > threshold:
            n = int(math.floor(duration / (2 * config.max_short_length)))
            if n <= 1:
                result.append(scene)
                continue

            part_len = duration / n
            for i in range(n):
                part_start = start_s + i * part_len
                part_end = start_s + (i + 1) * part_len
                result.append([_SecondsTime(part_start), _SecondsTime(part_end)])
        else:
            result.append(scene)

    return result


def find_smart_end_point(
    start_time: float,
    min_end: float,
    max_end: float,
    times: np.ndarray,
    scores: np.ndarray,
    search_window: float = 2.0
) -> float:
    """
    Search for the best end point (with minimal action/volume)
    in the range [max_end - search_window, max_end].
    If no good point is found, returns max_end.
    """
    # Ensure we search within the allowed range
    search_start = max(min_end, max_end - search_window)
    search_finish = max_end

    if search_finish <= search_start:
        return search_finish

    # Select score segment in the search area
    mask = (times >= search_start) & (times <= search_finish)
    if not np.any(mask):
        return search_finish

    t_seg = times[mask]
    s_seg = scores[mask]

    # Find index with minimum score value (silence/calmness)
    min_idx = np.argmin(s_seg)
    best_end_time = float(t_seg[min_idx])

    return best_end_time


def process_video(video_file: Path, config: ProcessingConfig, output_dir: Path) -> None:
    """Process a single video file and generate short clips."""

    logging.info("\nProcess: %s", video_file.name)

    logging.info("Detecting scenes (GPU)...")
    scene_list = detect_video_scenes_gpu(video_file)

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Computing audio action profile (GPU)...")
    audio_times, audio_score = compute_audio_action_profile(video_file)

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Computing video action profile (GPU)...")
    video_times, video_score = compute_video_action_profile(
        video_file,
        fps=4,
        downscale_factor=6,
    )

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-calculate video duration for boundary checks
    try:
        dmx = nvc.PyFFmpegDemuxer(str(video_file))
        video_duration = float(dmx.Numframes() / dmx.Framerate())
        del dmx
    except Exception:
        logging.warning("PyNvCodec probe failed, fallback to 0 duration.")
        video_duration = 0.0

    processed_scene_list = combine_scenes(scene_list, config)
    processed_scene_list = split_overlong_scenes(processed_scene_list, config)

    logging.info("Scenes list with action scores:")
    for i, scene in enumerate(processed_scene_list, start=1):
        duration = scene[1].get_seconds() - scene[0].get_seconds()
        score_val = scene_action_score(scene, audio_times, audio_score, video_times, video_score)
        logging.info(
            "    Scene %2d: Duration %5.1f s, ActionScore %7.3f,"
            " Start %s / Frame %d, End %s / Frame %d",
            i,
            duration,
            score_val,
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )

    sorted_processed_scene_list = sorted(
        processed_scene_list,
        key=lambda s: scene_action_score(s, audio_times, audio_score, video_times, video_score),
        reverse=True,
    )

    logging.info("Sorted scenes list (by action score):")
    for i, scene in enumerate(sorted_processed_scene_list, start=1):
        duration = scene[1].get_seconds() - scene[0].get_seconds()
        score_val = scene_action_score(scene, audio_times, audio_score, video_times, video_score)
        logging.info(
            "    Scene %2d: ActionScore %7.3f, Duration %5.1f s,"
            " Start %s / Frame %d, End %s / Frame %d",
            i,
            score_val,
            duration,
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )

    truncated_list = sorted_processed_scene_list[: config.scene_limit]

    if truncated_list:
        for i, scene in enumerate(truncated_list):
            scene_start = scene[0].get_seconds()
            scene_end = scene[1].get_seconds()
            scene_duration = scene_end - scene_start

            # STRATEGY 1: If scene fits entirely - take it all.
            # We add a small padding (1.5s) to capture the "end scene animation/fade".
            if scene_duration <= config.max_short_length:
                final_start = scene_start
                padding = 1.5
                final_end = min(scene_end + padding, video_duration)
                
                # Check if padding pushes us over max limit
                if (final_end - final_start) > config.max_short_length:
                    final_end = final_start + config.max_short_length
                
                final_duration = final_end - final_start
                logging.info(f"Scene {i}: Full scene + padding ({final_duration:.2f}s)")

            # STRATEGY 2: Scene too long, cut best window with smart end.
            else:
                target_duration = float(config.max_short_length)

                best_start = best_action_window_start(
                    scene,
                    target_duration,
                    audio_times,
                    audio_score,
                    video_times,
                    video_score,
                )

                absolute_min_end = best_start + config.min_short_length
                absolute_max_end = min(scene_end, best_start + config.max_short_length)

                final_end = find_smart_end_point(
                    best_start,
                    absolute_min_end,
                    absolute_max_end,
                    audio_times,
                    audio_score,
                    search_window=5.0
                )

                final_start = best_start
                final_duration = final_end - final_start
                logging.info(f"Scene {i}: Smart Cut. Start {final_start:.2f}, End {final_end:.2f} (Duration {final_duration:.2f}s)")

            render_file_name = f"{video_file.stem} scene-{i}{video_file.suffix}"
            render_path = output_dir / render_file_name

            # Prepare render params
            params = get_render_params(
                video_file,
                final_start,
                final_duration,
                config
            )

            # Execute GPU render
            render_video_gpu_isolated(
                params,
                render_path,
                max_error_depth=config.max_error_depth,
            )
    else:
        # No scenes found, fallback to random clip
        short_length = random.randint(
            config.min_short_length, config.max_short_length
        )

        if video_duration < config.max_short_length:
            adapted_short_length = min(math.floor(video_duration), short_length)
        else:
            adapted_short_length = short_length

        min_start_point = min(10, math.floor(video_duration) - adapted_short_length)
        max_start_point = math.floor(video_duration - adapted_short_length)

        start_point = float(random.randint(int(min_start_point), int(max_start_point)))

        params = get_render_params(
            video_file,
            start_point,
            float(adapted_short_length),
            config
        )

        render_video_gpu_isolated(
            params,
            output_dir / video_file.name,
            max_error_depth=config.max_error_depth,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the shorts generator."""
    parser = argparse.ArgumentParser(description="Generate short clips from gameplay footage using GPU.")
    return parser.parse_args()


def config_from_env() -> ProcessingConfig:
    """Build ProcessingConfig from environment variables."""
    return ProcessingConfig(
        target_ratio_w=_get_env_int("TARGET_RATIO_W", 1),
        target_ratio_h=_get_env_int("TARGET_RATIO_H", 1),
        scene_limit=_get_env_int("SCENE_LIMIT", 6),
        x_center=_get_env_float("X_CENTER", 0.5),
        y_center=_get_env_float("Y_CENTER", 0.5),
        max_error_depth=_get_env_int("MAX_ERROR_DEPTH", 3),
        min_short_length=_get_env_int("MIN_SHORT_LENGTH", 15),
        max_short_length=_get_env_int("MAX_SHORT_LENGTH", 179),
        max_combined_scene_length=_get_env_int("MAX_COMBINED_SCENE_LENGTH", 300),
    )


def main() -> None:
    """Entry point for command-line execution."""
    # args = parse_args()
    config = config_from_env()
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)

    gameplay_dir = Path("gameplay")
    if not gameplay_dir.exists():
         logging.warning("No 'gameplay' directory found. Exiting.")
         return

    for video_file in gameplay_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".mkv", ".mov"]:
            process_video(video_file, config, output_dir)


if __name__ == "__main__":
    main()
