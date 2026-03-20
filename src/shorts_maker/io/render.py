import gc
import logging
import math
import multiprocessing
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import PyNvCodec as nvc
import torch
from tqdm import tqdm

from shorts_maker.config import ProcessingConfig
from shorts_maker.io.streamer import GPUVideoStreamer

try:
    import resource
except ImportError:
    resource = None

logger = logging.getLogger(__name__)


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
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        usage_stats.append(f"VRAM Alloc: {allocated:.1f} MB, Res: {reserved:.1f} MB")

    logger.info(f"[{tag}] Memory: {', '.join(usage_stats)}")


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
    x = torch.nn.functional.pad(x, (radius, radius, radius, radius), mode="reflect")

    # Apply separable convolutions (Y first, then X)
    x = torch.nn.functional.conv2d(x, kernel_y, groups=channels)
    x = torch.nn.functional.conv2d(x, kernel_x, groups=channels)

    # Revert to original format
    if is_hwc:
        x = x.squeeze(0).permute(1, 2, 0)

    return x.to(image_tensor.dtype)


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
    del dmx

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
        bg_h = bg_w  # Force square output
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
        is_vertical_bg=is_vertical_bg,
    )


def render_video_gpu(
    params: RenderParams,
    output_path: Path,
    max_error_depth: int = 3,
) -> None:
    """Render the clip using GPU compositing and FFMPEG NVENC (Optimized)."""

    logger.info(f"Rendering GPU: {output_path.name}")

    # 1. Fast Extract Audio
    temp_audio = output_path.with_suffix(".aac")

    cmd_audio = [
        "/usr/bin/ffmpeg",
        "-y",
        "-ss",
        f"{params.start_time:.3f}",
        "-i",
        str(params.source_path),
        "-t",
        f"{params.duration:.3f}",
        "-q:a",
        "0",
        "-map",
        "a?",
        str(temp_audio),
    ]
    subprocess.run(
        cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

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
        "/usr/bin/ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{params.output_width}x{params.output_height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        f"{fps}",
        "-i",
        "-",
    ]

    has_valid_audio = temp_audio.exists() and temp_audio.stat().st_size > 0

    if has_valid_audio:
        cmd_ffmpeg.extend(["-i", str(temp_audio)])

    cmd_ffmpeg.extend(
        [
            "-c:v",
            "hevc_nvenc",  # Use hardware encoder
            "-preset",
            "slow",  # 'slow' is compatible with old and new ffmpeg NVENC
            "-rc",
            "vbr",
            "-cq",
            "23",  # Slightly increased CQ to reduce bitrate spikes
            "-maxrate",
            "80M",  # Cap bitrate to prevent buffer bloat
            "-bufsize",
            "100M",
            "-pix_fmt",
            "yuv420p",
            "-g",
            f"{int(fps * 2)}",
            "-bf",
            "2",
        ]
    )

    if has_valid_audio:
        cmd_ffmpeg.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd_ffmpeg.extend(["-shortest", str(output_path)])

    # redirect stderr to a file to prevent buffer deadlock
    log_path = output_path.with_suffix(".ffmpeg.log")
    ffmpeg_log = open(log_path, "w")
    
    process = None

    try:
        process = subprocess.Popen(
            cmd_ffmpeg,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log,
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
            del dmx

            if params.is_vertical_bg:
                bg_ratio_w, bg_ratio_h = 9, 16
                if (src_w / src_h) > (bg_ratio_w / bg_ratio_h):
                    bg_crop_h, bg_crop_w = src_h, int(src_h * bg_ratio_w / bg_ratio_h)
                else:
                    bg_crop_w, bg_crop_h = src_w, int(src_w * bg_ratio_h / bg_ratio_w)
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(
                    src_h * 0.5 - bg_crop_h / 2
                )
            else:
                bg_dim = min(src_w, src_h)
                bg_crop_w, bg_crop_h = bg_dim, bg_dim
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(
                    src_h * 0.5 - bg_crop_h / 2
                )

            total_frames = int(params.duration * fps)

            BATCH_SIZE = 4
            total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE

            log_memory_usage("Render Start")

            with tqdm(
                total=total_batches, desc="Video render", unit="batch"
            ) as pbar_render, GPUVideoStreamer(
                params.source_path, seek_time=params.start_time
            ) as streamer:

                batch_count = 0
                for frames, _ in streamer.stream_batches(
                    batch_size=BATCH_SIZE, max_frames=total_frames
                ):
                    if batch_count % 50 == 0:
                        logger.info(f"Rendering batch {batch_count}/{total_batches}")
                    batch_count += 1

                    if process.poll() is not None:
                        logger.error("FFMPEG died")
                        break

                    # 1. Background Processing
                    bg_frames = frames[
                        :,
                        bg_crop_y : bg_crop_y + bg_crop_h,
                        bg_crop_x : bg_crop_x + bg_crop_w,
                        :,
                    ]
                    bg_frames = bg_frames.permute(0, 3, 1, 2).float()  # to NCHW

                    # Resize for blur (low res)
                    blur_w, blur_h = 720, (1280 if params.is_vertical_bg else 720)
                    bg_small = torch.nn.functional.interpolate(
                        bg_frames,
                        size=(blur_h, blur_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # Blur via Native PyTorch (bg_small is already NCHW)
                    blurred_bg = blur_gpu(bg_small, sigma=16.0)
                    final_bg = torch.nn.functional.interpolate(
                        blurred_bg,
                        size=(params.output_height, params.output_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # 2. Foreground Processing
                    fg_frames = frames[
                        :,
                        params.crop_y : params.crop_y + params.crop_h,
                        params.crop_x : params.crop_x + params.crop_w,
                        :,
                    ]
                    fg_frames = fg_frames.permute(0, 3, 1, 2).float()
                    final_fg = torch.nn.functional.interpolate(
                        fg_frames, size=(fg_h, fg_w), mode="bilinear", align_corners=False
                    )

                    # 3. Composite (Overlay)
                    y_off, x_off = (params.output_height - fg_h) // 2, (
                        params.output_width - fg_w
                    ) // 2
                    y1, y2 = max(0, y_off), min(params.output_height, y_off + fg_h)
                    x1, x2 = max(0, x_off), min(params.output_width, x_off + fg_w)
                    sy1, sx1 = max(0, -y_off), max(0, -x_off)

                    # Direct tensor insertion
                    if (y2 > y1) and (x2 > x1):
                        final_bg[:, :, y1:y2, x1:x2] = final_fg[
                            :, :, sy1 : (sy1 + (y2 - y1)), sx1 : (sx1 + (x2 - x1))
                        ]

                    # 4. Write to Pipe
                    # Convert to byte and move to CPU
                    out_tensor = final_bg.permute(0, 2, 3, 1).contiguous().byte()
                    out_bytes = out_tensor.cpu().numpy().tobytes()

                    try:
                        import select

                        _, writable, _ = select.select(
                            [], [process.stdin.fileno()], [], 10.0
                        )
                        if not writable:
                            logger.error("FFMPEG stdin write blocked (timeout)")
                            break
                        process.stdin.write(out_bytes)
                    except BrokenPipeError:
                        break

                    # 5. Explicit Cleanup (Critical for Loop)
                    del (
                        frames,
                        bg_frames,
                        bg_small,
                        blurred_bg,
                        final_bg,
                        fg_frames,
                        final_fg,
                        out_tensor,
                        out_bytes,
                    )

                    # Periodic GC
                    if batch_count > 0 and batch_count % 100 == 0:
                        gc.collect()

                    pbar_render.update(1)

        except Exception as e:
            logger.error(f"Error during GPU render: {e}", exc_info=True)
        finally:
            # Clean up processes and memory
            if process:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.wait()

            if "ffmpeg_log" in locals() and ffmpeg_log:
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
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=render_video_gpu, args=args, kwargs=kwargs)
    p.start()
    p.join()

    if p.exitcode != 0:
        logger.error("Render process failed with exit code %s", p.exitcode)
        if p.exitcode == -9 or p.exitcode == 137:
            logger.error("Render process was likely OOM killed.")
