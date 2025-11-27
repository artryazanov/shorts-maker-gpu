"""Utility for automatically generating short video clips.

This module processes gameplay videos and creates resized clips
that fit common short-video aspect ratios. Scene detection is used
to select interesting parts of the video and background blurring is
applied when required.

The script was refactored to improve readability and maintainability
while retaining the original behaviour.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from moviepy import CompositeVideoClip, VideoFileClip
from scipy.ndimage import gaussian_filter
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
import librosa


# Load environment variables from a .env file if present.
load_dotenv()

# Configure basic logging. The calling application may override this
# configuration if a different format is required.
logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def detect_video_scenes(video_path: Path, threshold: float = 27.0) -> Sequence[Tuple] | List:
    """Detect scenes in the provided video file.

    Parameters
    ----------
    video_path: Path
        Path to the video file.
    threshold: float, optional
        Threshold value for the ``ContentDetector``.

    Returns
    -------
    Sequence[Tuple]
        List of ``(start, end)`` timecodes for each detected scene.
    """

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    return scene_manager.get_scene_list()


def blur(image: np.ndarray) -> np.ndarray:
    """Return a blurred version of ``image``."""
    return gaussian_filter(image.astype(float), sigma=8)


# --- Audio-based action scoring -------------------------------------------------

def compute_audio_action_profile(
    video_path: Path,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute audio-based "action score" over the entire video.

    Returns:
      times  - array of times (seconds) for each feature frame
      score  - combined action score (loudness + spectral "roughness")
    """

    # librosa can read audio directly from mp4
    y, sr = librosa.load(str(video_path), sr=None, mono=True)

    # RMS (loudness)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]  # shape: (n_frames,)

    # Spectral flux (how much the spectrum changes from frame to frame)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    spectral_flux = np.sqrt(((np.diff(S, axis=1) ** 2).sum(axis=0)))
    spectral_flux = np.concatenate([[0.0], spectral_flux])  # align length

    def smooth(x: np.ndarray, win: int = 15) -> np.ndarray:
        # Ensure smoothing window does not exceed the signal length to avoid
        # np.convolve(..., mode="same") expanding the output when win > len(x).
        n = max(1, min(int(win), int(len(x))))
        if n == 1:
            return x
        kernel = np.ones(n, dtype=float) / float(n)
        return np.convolve(x, kernel, mode="same")

    # Normalization
    rms_norm = (rms - rms.mean()) / (rms.std() + 1e-8)
    flux_norm = (spectral_flux - spectral_flux.mean()) / (spectral_flux.std() + 1e-8)

    # Smoothing to remove jitter
    rms_smooth = smooth(rms_norm, win=21)
    flux_smooth = smooth(flux_norm, win=21)

    # Final score: tweak weights if needed
    score = 0.6 * rms_smooth + 0.4 * flux_smooth

    # Time for each feature frame
    times = librosa.frames_to_time(
        np.arange(len(score)),
        sr=sr,
        hop_length=hop_length,
    )

    return times, score



def compute_video_action_profile(
    video_path: Path,
    fps: int = 6,
    downscale_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a fast video-based "action score" over the entire video.

    Optimizations:
      - Read frames sequentially via iter_frames(...) at a low analysis fps.
      - Downscale frames by taking every N-th pixel along each axis to reduce cost.
      - Compute mean absolute difference in grayscale luma between consecutive frames.
      - Normalize and smooth similarly to the audio profile.

    Returns:
      times  - array of timestamps (seconds) for each sample
      score  - normalized video action score (higher means more motion)
    """

    clip = VideoFileClip(str(video_path))
    duration = float(clip.duration)

    if duration <= 0:
        clip.close()
        return np.array([], dtype=float), np.array([], dtype=float)

    # Do not analyze faster than the source fps.
    orig_fps = clip.fps or fps
    eff_fps = min(float(fps), float(orig_fps))
    if eff_fps <= 0:
        eff_fps = max(1.0, float(fps))

    motions: List[float] = []
    times: List[float] = []

    prev_gray: np.ndarray | None = None

    frame_iter = clip.iter_frames(fps=eff_fps, dtype="uint8", logger="bar")

    for idx, frame in enumerate(frame_iter):
        t = idx / eff_fps
        if t > duration:
            break

        # Convert to grayscale [0, 1]
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0

        # Downscale: simple striding is sufficient to estimate motion.
        if downscale_factor > 1:
            gray = gray[::downscale_factor, ::downscale_factor]

        if prev_gray is None:
            motions.append(0.0)
        else:
            diff = np.mean(np.abs(gray - prev_gray))
            motions.append(float(diff))

        prev_gray = gray
        times.append(t)

    clip.close()

    if not motions:
        return np.array([], dtype=float), np.array([], dtype=float)

    motions = np.asarray(motions, dtype=float)
    times_arr = np.asarray(times, dtype=float)

    # Normalization (z-score)
    motions_norm = (motions - motions.mean()) / (motions.std() + 1e-8)

    def smooth(x: np.ndarray, win: int = 15) -> np.ndarray:
        n = max(1, min(int(win), int(len(x))))
        if n == 1:
            return x
        kernel = np.ones(n, dtype=float) / float(n)
        return np.convolve(x, kernel, mode="same")

    # Smooth over roughly ~1 second (window ≈ eff_fps samples)
    score = smooth(motions_norm, win=int(eff_fps))

    return times_arr, score


def scene_action_score(
    scene: Tuple,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Return total (summed) action score within the scene.

    Now accounts for both audio and video:
      total = w_audio * audio_action + w_video * video_action
    """

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
        # If video profile wasn't computed
        return audio_val

    video_val = _segment_sum(video_times, video_score)

    return w_audio * audio_val + w_video * video_val


def _best_window_single(
    scene: Tuple,
    window_length: float,
    times: np.ndarray,
    score: np.ndarray,
) -> float:
    """Audio OR video variant of best_action_window_start for a single profile.

    Returns the best start time within the scene that maximizes the sum of
    `score` over a sliding window of length `window_length` using the
    timestamps `times`. Falls back to the scene start in degenerate cases.
    """

    start_sec = float(scene[0].get_seconds())
    end_sec = float(scene[1].get_seconds())

    # Safety clamp
    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return start_sec

    max_allowed_start = end_sec - float(window_length)
    if max_allowed_start <= start_sec:
        # Window almost equals scene length — start at scene start
        return max(start_sec, min(start_sec, end_sec - float(window_length)))

    # Keep only samples inside the scene
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

    # Window length in samples
    n_win = int(max(1, round(float(window_length) / dt)))
    if len(s_seg) < n_win:
        return start_sec

    # Fast moving sum via cumulative sum
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
    """Find the start of the window inside the scene maximizing combined action.

    - If both audio and video are present → combine on the audio grid:
        combined = w_audio * audio + w_video * video_interp
      (video_interp is the video score interpolated to `audio_times`).
    - If video is missing/invalid → fallback to audio-only behavior.
    - If audio is not suitable but video exists → fallback to video-only.
    """

    # If video is missing or empty, use audio-only profile as before
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

    # Use audio samples inside the scene as the base grid
    a_mask = (audio_times >= start_sec) & (audio_times <= end_sec)
    if not np.any(a_mask):
        # If audio has no samples but video exists → try video-only
        return _best_window_single(scene, window_length, video_times, video_score)

    t_a_seg = audio_times[a_mask]
    s_a_seg = audio_score[a_mask]

    if len(t_a_seg) < 2:
        # Too few audio samples → try video-only
        return _best_window_single(scene, window_length, video_times, video_score)

    # Interpolate video to audio timestamps
    if len(video_times) > 1:
        order = np.argsort(video_times)
        v_interp = np.interp(t_a_seg, video_times[order], video_score[order])
    else:
        v_interp = np.full_like(t_a_seg, float(video_score[0]), dtype=float)

    combined_seg = w_audio * s_a_seg + w_video * v_interp

    # Proceed like in _best_window_single but on (t_a_seg, combined_seg)
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


def crop_clip(
    clip: VideoFileClip,
    ratio_w: int,
    ratio_h: int,
    x_center: float,
    y_center: float,
):
    """Crop ``clip`` to the desired aspect ratio.

    The centre of the crop is determined by ``x_center`` and ``y_center``,
    which are expressed as fractions of the clip's width and height.
    """

    width, height = clip.size
    current_ratio = width / height
    target_ratio = ratio_w / ratio_h

    if current_ratio > target_ratio:
        new_width = round(height * ratio_w / ratio_h)
        return clip.cropped(
            width=new_width,
            height=height,
            x_center=width * x_center,
            y_center=height * y_center,
        )

    new_height = round(width / ratio_w * ratio_h)
    return clip.cropped(
        width=width,
        height=new_height,
        x_center=width * x_center,
        y_center=height * y_center,
    )


def render_video(
    clip: VideoFileClip,
    video_file_name: Path,
    output_dir: Path,
    depth: int = 0,
    max_error_depth: int = 3,
) -> None:
    """Render ``clip`` to ``output_dir``

    Parameters
    ----------
    clip:
        The video clip to render.
    video_file_name:
        Name of the output file.
    output_dir:
        Directory where the output will be written.
    depth:
        Current retry depth used to limit recursion.
    max_error_depth:
        Maximum number of retries permitted before surfacing an error.
    """

    try:
        clip.write_videofile(
            str(output_dir / video_file_name.name),
            codec="libx264",
            audio_codec="aac",
            fps=min(getattr(clip, "fps", 60), 60),
        )
    except Exception:  # pragma: no cover - logging only
        if depth < max_error_depth:
            logging.exception("Rendering failed, retrying...")
            render_video(
                clip,
                video_file_name,
                output_dir,
                depth + 1,
                max_error_depth,
            )
        else:
            logging.exception("Rendering failed after multiple attempts.")
            raise


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


def get_final_clip(
    clip: VideoFileClip,
    start_point: float,
    final_clip_length: float,
    config: ProcessingConfig,
) -> VideoFileClip:
    """Prepare a clip ready for rendering."""

    result_clip = clip.subclipped(start_point, start_point + final_clip_length)

    width, height = result_clip.size
    target_ratio = config.target_ratio_w / config.target_ratio_h
    if width / height > target_ratio:
        result_clip = crop_clip(
            result_clip,
            config.target_ratio_w,
            config.target_ratio_h,
            config.x_center,
            config.y_center,
        )

    width, height = result_clip.size
    bg_w, bg_h = select_background_resolution(width)
    result_clip = result_clip.resized(width=bg_w)

    if width >= height:
        background_clip = clip.subclipped(start_point, start_point + final_clip_length)
        background_clip = crop_clip(background_clip, 1, 1, config.x_center, config.y_center)
        background_clip = background_clip.resized(width=720, height=720)
        background_clip = background_clip.image_transform(blur)
        background_clip = background_clip.resized(width=bg_w, height=bg_w)
        result_clip = CompositeVideoClip([background_clip, result_clip.with_position("center")])
    elif width / 9 < height / 16:
        background_clip = clip.subclipped(start_point, start_point + final_clip_length)
        background_clip = crop_clip(background_clip, 9, 16, config.x_center, config.y_center)
        background_clip = background_clip.resized(width=720, height=1280)
        background_clip = background_clip.image_transform(blur)
        background_clip = background_clip.resized(width=bg_w, height=bg_h)
        result_clip = CompositeVideoClip([background_clip, result_clip.with_position("center")])

    return result_clip


def combine_scenes(scene_list: Sequence[Tuple], config: ProcessingConfig) -> List[List]:
    """Combine adjacent scenes while preserving content.

    Key principles:
    - Never drop interior content just because a run is shorter than a mid target.
    - Prefer to merge short interior runs into neighbouring runs.
    - Only drop too-short runs that are at the very beginning or end (boundaries),
      matching the original test expectations.
    - For long sequences of short scenes, cap chunks around `max_combined_scene_length`.
    """

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

            # If it's a short-scenes run that gets very long, flush it.
            if run_type_small:
                run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
                if run_duration > config.max_combined_scene_length:
                    # Exceeded cap: flush up to the end of the previous scene to avoid overlap
                    prev_end_time = scene_list[i - 1][1]
                    out.append([run_start_time, prev_end_time])
                    # Start a new run from current scene
                    run_start_idx = i
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                elif run_duration == config.max_combined_scene_length:
                    is_last_scene = (i == n - 1)
                    if is_last_scene:
                        # At the very end, close at previous boundary so the final tiny tail
                        # (current scene) remains a boundary run which can be dropped by threshold.
                        prev_end_time = scene_list[i - 1][1]
                        out.append([run_start_time, prev_end_time])
                        run_start_idx = i
                        run_start_time = scene_list[i][0]
                        run_end_time = scene_list[i][1]
                    else:
                        # Exactly at cap and not the last scene: we can safely include current scene
                        # to reach the cap precisely.
                        out.append([run_start_time, run_end_time])
                        # Start new run at the next scene. Its start equals current end.
                        run_start_idx = i + 1
                        run_start_time = scene_list[i][1]
                        run_end_time = scene_list[i][1]
        else:
            # Run ends at i-1; decide how to handle it.
            run_end_idx = i - 1
            run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
            is_boundary = (run_start_idx == 0) or (run_end_idx == n - 1)
            threshold = config.middle_short_length if is_boundary else config.min_short_length

            if run_duration >= threshold:
                out.append([run_start_time, run_end_time])
                # Start a new run at i
                run_start_idx = i
                run_type_small = current_small
                run_start_time = scene_list[i][0]
                run_end_time = scene_list[i][1]
            else:
                # Too short run.
                if is_boundary and run_start_idx == 0:
                    # At the very start: drop this head run (keep original behavior)
                    run_start_idx = i
                    run_type_small = current_small
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                else:
                    # Interior: merge with the next run by carrying the start forward.
                    run_type_small = current_small
                    run_end_time = scene_list[i][1]
                    # Note: keep run_start_idx/time unchanged to include previous run.

    # Flush the final run (boundary)
    final_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
    is_boundary = True  # the last run always reaches the end
    threshold = config.middle_short_length if is_boundary else config.min_short_length
    if final_duration >= threshold:
        out.append([run_start_time, run_end_time])

    return out


class _SecondsTime:
    """Lightweight stand-in for scene time objects using seconds.

    Provides the minimal API used elsewhere: get_seconds(), get_timecode(), get_frames().
    Frames are computed assuming 30 fps to keep behavior consistent with tests.
    """

    def __init__(self, seconds: float):
        self._seconds = float(seconds)

    def get_seconds(self) -> float:
        return self._seconds

    def get_timecode(self) -> str:
        # Keep simple representation similar to tests' MockTime
        return str(self._seconds)

    def get_frames(self) -> int:
        return int(self._seconds * 30)


def split_overlong_scenes(combined_scene_list: List[List], config: ProcessingConfig) -> List[List]:
    """Split scenes longer than 4 * max_short_length into n equal parts.

    For each scene with duration D > 4 * max_short_length, compute
    n = floor(D / (2 * max_short_length)) and split the scene into n
    equal sub-scenes. Scenes not exceeding the threshold are kept as is.
    """

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


def process_video(video_file: Path, config: ProcessingConfig, output_dir: Path) -> None:
    """Process a single video file and generate short clips."""

    logging.info("\nProcess: %s", video_file.name)

    logging.info("Detecting scenes...")
    scene_list = detect_video_scenes(video_file)

    logging.info("Computing audio action profile...")
    audio_times, audio_score = compute_audio_action_profile(video_file)

    logging.info("Computing video action profile...")
    video_times, video_score = compute_video_action_profile(
        video_file,
        fps=4,                # lower analysis fps for speed (3–6 is a good range)
        downscale_factor=6,   # strong spatial downscale (4–8) for faster motion estimation
    )

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

    # Sort by action score, not by length
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

    video_clip = VideoFileClip(str(video_file))
    truncated_list = sorted_processed_scene_list[: config.scene_limit]

    logging.info("Truncated sorted scenes list:")
    for i, scene in enumerate(truncated_list, start=1):
        logging.info(
            "    Scene %2d: Duration %d Start %s / Frame %d, End %s / Frame %d",
            i,
            scene[1].get_seconds() - scene[0].get_seconds(),
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )

    if truncated_list:
        for i, scene in enumerate(truncated_list):
            duration = math.floor(scene[1].get_seconds() - scene[0].get_seconds())
            short_length = random.randint(
                config.min_short_length, min(config.max_short_length, duration)
            )

            # Pick the start time that maximizes the cumulative audio action
            # within the chosen short_length window for this scene.
            best_start = best_action_window_start(
                scene,
                float(short_length),
                audio_times,
                audio_score,
                video_times,
                video_score,
            )
            logging.info(
                "Selected start %.2f for scene %d with window %ds",
                best_start,
                i,
                short_length,
            )

            final_clip = get_final_clip(
                video_clip,
                best_start,
                short_length,
                config,
            )

            render_file_name = f"{video_file.stem} scene-{i}{video_file.suffix}"
            render_video(
                final_clip,
                Path(render_file_name),
                output_dir,
                max_error_depth=config.max_error_depth,
            )
    else:
        short_length = random.randint(
            config.min_short_length, config.max_short_length
        )

        if video_clip.duration < config.max_short_length:
            adapted_short_length = min(math.floor(video_clip.duration), short_length)
        else:
            adapted_short_length = short_length

        min_start_point = min(10, math.floor(video_clip.duration) - adapted_short_length)
        max_start_point = math.floor(video_clip.duration - adapted_short_length)
        final_clip = get_final_clip(
            video_clip,
            random.randint(min_start_point, max_start_point),
            adapted_short_length,
            config,
        )
        render_video(
            final_clip,
            video_file,
            output_dir,
            max_error_depth=config.max_error_depth,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the shorts generator."""

    parser = argparse.ArgumentParser(description="Generate short clips from gameplay footage.")
    return parser.parse_args()


def config_from_env() -> ProcessingConfig:
    """Build ProcessingConfig from environment variables with sane defaults.

    Environment variables:
      - TARGET_RATIO_W (int)
      - TARGET_RATIO_H (int)
      - SCENE_LIMIT (int)
      - X_CENTER (float)
      - Y_CENTER (float)
      - MAX_ERROR_DEPTH (int)
      - MIN_SHORT_LENGTH (int)
      - MAX_SHORT_LENGTH (int)
      - MAX_COMBINED_SCENE_LENGTH (int)
    """

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

    args = parse_args()
    config = config_from_env()
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)

    gameplay_dir = Path("gameplay")
    for video_file in gameplay_dir.iterdir():
        if video_file.is_file():
            process_video(video_file, config, output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

