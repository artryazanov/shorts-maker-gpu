import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        logger.error(f"Failed to load audio from {video_path}")
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
    pbar = tqdm(
        total=total_samples if total_samples > 0 else 1,
        desc="Audio Profile",
        unit="samples",
    )

    while current_frame < total_samples or total_samples <= 0:
        read_count = chunk_frames + (overlap_frames if current_frame > 0 else 0)
        read_start = max(0, current_frame - overlap_frames)

        try:
            waveform, sr = torchaudio.load(
                str(video_path),
                frame_offset=read_start,
                num_frames=read_count,
                normalize=True,
            )
        except Exception:
            logger.error(f"Error reading audio chunk at {read_start}")
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
        y_padded = torch.nn.functional.pad(
            chunk_tensor.unsqueeze(0), (2048 // 2, 2048 // 2), mode="reflect"
        ).squeeze(0)
        stft_chunk = torch.stft(
            y_padded,
            n_fft=2048,
            hop_length=hop_length,
            window=window,
            center=False,
            return_complex=True,
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
    spectral_flux = (
        torch.cat(flux_values) if flux_values else torch.tensor([], device=device)
    )

    # --- Post Processing ---
    min_len = min(rms.shape[0], spectral_flux.shape[0])
    rms = rms[:min_len]
    spectral_flux = spectral_flux[:min_len]

    rms_mean = rms.mean() if rms.numel() > 0 else torch.tensor(0.0, device=device)
    rms_std = (
        (rms.std() + 1e-8) if rms.numel() > 0 else torch.tensor(1.0, device=device)
    )
    rms_norm = (rms - rms_mean) / rms_std if rms.numel() > 0 else rms

    flux_mean = (
        spectral_flux.mean()
        if spectral_flux.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    flux_std = (
        (spectral_flux.std() + 1e-8)
        if spectral_flux.numel() > 0
        else torch.tensor(1.0, device=device)
    )
    flux_norm = (
        (spectral_flux - flux_mean) / flux_std
        if spectral_flux.numel() > 0
        else spectral_flux
    )

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

    score = (
        0.6 * rms_smooth + 0.4 * flux_smooth
        if rms_smooth.numel() > 0 and flux_smooth.numel() > 0
        else (rms_smooth if flux_smooth.numel() == 0 else flux_smooth)
    )

    num_frames_out = score.shape[0]
    times = (
        torch.arange(num_frames_out, device=device) * hop_length / sample_rate
        if num_frames_out > 0
        else torch.tensor([], device=device)
    )

    return times.cpu().numpy(), score.cpu().numpy()
