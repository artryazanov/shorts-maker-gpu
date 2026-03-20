import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import PyNvCodec as nvc
import torch
from tqdm import tqdm

from shorts_maker.io.streamer import GPUVideoStreamer

logger = logging.getLogger(__name__)


@torch.no_grad()  # type: ignore
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
        logger.warning("Failed to load video for action profile.", exc_info=True)
        return np.array([]), np.array([])

    eff_fps = min(float(fps), orig_fps)
    if eff_fps <= 0:
        eff_fps = max(1.0, float(fps))

    # Calculate step for subsampling
    step = max(1, int(orig_fps / eff_fps))

    motions = []
    times = []
    prev_batch_last = None

    with GPUVideoStreamer(
        video_path, target_width=w_new, target_height=h_new
    ) as streamer:
        total_batches = int(np.ceil(streamer.total_frames / (step * 16)))
        pbar = tqdm(total=total_batches, desc="Video Action Profile", unit="batch")

        # GPUVideoStreamer natively handles iterating to the end without hanging
        # and outputs only the batches representing the requested `step`
        for frames_subset, global_indices in streamer.stream_batches(
            batch_size=16, step=step
        ):
            frames_subset = frames_subset.float()

            # Grayscale conversion on GPU
            gray = (
                frames_subset[..., 0] * 0.299
                + frames_subset[..., 1] * 0.587
                + frames_subset[..., 2] * 0.114
            )

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

    motions_t = torch.cat(motions)
    times_t = torch.cat(times)

    # Normalize and smooth (similar to audio)
    if motions_t.numel() == 0:
        return np.array([]), np.array([])
    if motions_t.std() == 0:
        motions_norm = motions_t
    else:
        motions_norm = (motions_t - motions_t.mean()) / (motions_t.std() + 1e-8)

    # Smooth
    def smooth_gpu(x: torch.Tensor, win: int) -> torch.Tensor:
        if win > x.shape[0]:
            win = x.shape[0]
        if win < 2:
            return x
        kernel = torch.ones(win, device=x.device) / win
        x_reshaped = x.view(1, 1, -1)
        kernel_reshaped = kernel.view(1, 1, -1)
        out = torch.nn.functional.conv1d(x_reshaped, kernel_reshaped, padding=win // 2)
        return out.view(-1)[: x.shape[0]]

    score = smooth_gpu(motions_norm, win=int(eff_fps))

    return times_t.cpu().numpy(), score.cpu().numpy()
