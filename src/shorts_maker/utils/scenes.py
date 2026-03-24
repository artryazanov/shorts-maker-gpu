"""Module for detecting scenes, calculating smart boundaries, and evaluating action density."""

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import PyNvCodec as nvc
from tqdm import tqdm

from shorts_maker.config import ProcessingConfig
from shorts_maker.io.streamer import GPUVideoStreamer


class _SecondsTime:
    """Lightweight stand-in for scene time objects using seconds.

    Acts as a wrapper for time values to maintain compatibility with legacy 
    or external scene detection logic without importing heavy dependencies.

    Attributes:
        _seconds (float): The exact time in seconds.
    """

    def __init__(self, seconds: float):
        self._seconds = float(seconds)

    def get_seconds(self) -> float:
        """Returns the time in seconds."""
        return self._seconds

    def get_timecode(self) -> str:
        """Returns the time formatted as a string with two decimal places."""
        return f"{self._seconds:.2f}"

    def get_frames(self) -> int:
        """Converts seconds to frames assuming a fixed 30 FPS target."""
        return int(self._seconds * 30)


def detect_video_scenes_gpu(
    video_path: Path | str, threshold: float = 27.0
) -> List[Tuple[_SecondsTime, _SecondsTime]]:
    """Detects scene changes in a video using GPU-accelerated I/O.

    This implementation replicates the exact semantics of PySceneDetect's 
    ContentDetector (v0.6.7) to produce identical scene cuts, but leverages 
    VPF (GPUVideoStreamer) for drastically faster frame extraction and resizing.
    It calculates the mean absolute difference between adjacent frames in HSV space.

    Args:
        video_path: Path to the input video file.
        threshold: The threshold for the frame score to trigger a scene cut.
            Higher values require more visual change to trigger a cut.

    Returns:
        A list of tuples, where each tuple represents a scene containing 
        the start time and end time as `_SecondsTime` objects.
    """
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
            return self._filter_length  # pragma: no cover

        def filter(self, frame_num: int, above_threshold: bool) -> List[int]:
            if not (self._filter_length > 0):  # pragma: no cover
                return [frame_num] if above_threshold else []  # pragma: no cover
            if self._last_above is None:
                self._last_above = frame_num
            # MERGE path
            return self._filter_merge(frame_num, above_threshold)

        def _filter_merge(self, frame_num: int, above_threshold: bool) -> List[int]:
            assert self._last_above is not None
            min_length_met = (frame_num - self._last_above) >= self._filter_length
            if above_threshold:
                self._last_above = frame_num
            if self._merge_triggered:
                assert self._merge_start is not None  # pragma: no cover
                num_merged_frames = self._last_above - self._merge_start  # pragma: no cover
                if min_length_met and (not above_threshold) and (num_merged_frames >= self._filter_length):  # pragma: no cover
                    self._merge_triggered = False  # pragma: no cover
                    return [self._last_above]  # pragma: no cover
                return []  # pragma: no cover
            if not above_threshold:
                return []
            if min_length_met:
                self._merge_enabled = True
                return [frame_num]
            if self._merge_enabled:  # pragma: no cover
                self._merge_triggered = True  # pragma: no cover
                self._merge_start = frame_num  # pragma: no cover
            return []  # pragma: no cover

    min_scene_len = int(fps * 1.5)
    if min_scene_len < 15:
        min_scene_len = 15
    flash_filter = _FlashFilterMerge(length=min_scene_len)

    # 4) Iterate frames, compute HSV components & frame score like ContentDetector
    batch_size = 16
    total_batches = (frame_count + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Detect scenes", unit="batch")

    last_hsv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    cut_indices: List[int] = []

    with GPUVideoStreamer(
        video_path,
        target_width=w_eff,
        target_height=h_eff,
        pix_fmt=nvc.PixelFormat.BGR,
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
                    above = False
                    flash_filter.filter(frame_num, above_threshold=above)
                    continue

                hue_prev, sat_prev, val_prev = last_hsv
                # Mean pixel distance per channel (match _mean_pixel_distance)
                # cast to int32 to avoid uint8 underflow
                dh = np.abs(hue.astype(np.int32) - hue_prev.astype(np.int32)).sum() / float(hue.size)
                ds = np.abs(sat.astype(np.int32) - sat_prev.astype(np.int32)).sum() / float(sat.size)
                dv = np.abs(val.astype(np.int32) - val_prev.astype(np.int32)).sum() / float(val.size)
                frame_score = (dh + ds + dv) / 3.0

                last_hsv = (hue, sat, val)

                # Compare against threshold exactly like ContentDetector
                above = frame_score >= threshold
                emitted = flash_filter.filter(frame_num=frame_num, above_threshold=above)
                if emitted:
                    cut_indices.extend(emitted)

            pbar.update(1)
            del frames_bgr, frames_cpu

    pbar.close()

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


def scene_action_score(
    scene: Tuple[_SecondsTime, _SecondsTime],
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Calculates the total integrated action score for a given scene.

    Integrates the combined audio and video action scores over the duration 
    of the scene to determine how "interesting" or action-packed the scene is.

    Args:
        scene: Tuple representing (start_time, end_time).
        audio_times: Array of timestamps for audio scores.
        audio_score: Array of computed audio action scores.
        video_times: Array of timestamps for video scores (optional).
        video_score: Array of computed video action scores (optional).
        w_audio: Weight modifier for the audio score (default: 0.6).
        w_video: Weight modifier for the video score (default: 0.4).

    Returns:
        The total sum of action scores falling within the scene boundaries.
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
        return audio_val

    video_val = _segment_sum(video_times, video_score)

    return w_audio * audio_val + w_video * video_val


def _best_window_single(
    scene: Tuple[_SecondsTime, _SecondsTime],
    window_length: float,
    times: np.ndarray,
    score: np.ndarray,
) -> float:
    """Internal helper to find the optimal start point using a single action profile."""
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
    scene: Tuple[_SecondsTime, _SecondsTime],
    window_length: float,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Finds the optimal starting point for a clip inside a long scene.

    Uses a sliding window approach over the combined audio/video action signals
    to maximize the amount of action captured within the requested window length.
    Automatically interpolates video scores to match the higher frequency audio timeline.

    Args:
        scene: Tuple representing the full scene boundaries.
        window_length: Desired duration of the final short clip (in seconds).
        audio_times: Timestamps of audio features.
        audio_score: Values of audio features.
        video_times: Timestamps of video features.
        video_score: Values of video features.
        w_audio: Weight given to audio action.
        w_video: Weight given to video action.

    Returns:
        The optimal timestamp (in seconds) to start the clip.
    """
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
        v_interp = np.full_like(t_a_seg, float(video_score[0]), dtype=float)  # pragma: no cover

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


def combine_scenes(
    scene_list: Sequence[Tuple[_SecondsTime, _SecondsTime]], config: ProcessingConfig
) -> List[Tuple[_SecondsTime, _SecondsTime]]:
    """Merges fragmented, adjacent micro-scenes into cohesive segments.

    Prevents the generation of overly fragmented clips by combining consecutive 
    scenes that fall below the `min_short_length` threshold defined in the config.

    Args:
        scene_list: Raw sequence of detected scenes.
        config: Settings controlling minimum and maximum scene lengths.

    Returns:
        A condensed list of scene boundaries.
    """
    if not scene_list:
        return []

    out: List[Tuple[_SecondsTime, _SecondsTime]] = []
    current_start = scene_list[0][0]
    current_end = scene_list[0][1]

    for i in range(1, len(scene_list)):
        duration = current_end.get_seconds() - current_start.get_seconds()
        
        # If the accumulated segment is smaller than the minimum, attach the next scene to it
        if duration < config.min_short_length:
            current_end = scene_list[i][1]
        else:
            # The segment is large enough, save it as a complete scene
            out.append((current_start, current_end))
            # Start tracking the new scene
            current_start = scene_list[i][0]
            current_end = scene_list[i][1]

    # Process the final remaining "tail" segment
    final_duration = current_end.get_seconds() - current_start.get_seconds()
    if final_duration >= config.min_short_length:
        out.append((current_start, current_end))
    elif out:
        # If the tail is too short but there were previous scenes - attach it to the last one
        prev_start, _ = out.pop()
        out.append((prev_start, current_end))
    elif final_duration > 0:
        # If this is the only scene in the entire video and it's short
        out.append((current_start, current_end))

    return out


def split_overlong_scenes(
    combined_scene_list: List[Tuple[_SecondsTime, _SecondsTime]], config: ProcessingConfig
) -> List[Tuple[_SecondsTime, _SecondsTime]]:
    """Splits extremely long scenes into manageable, equally-sized chunks.

    Prevents the rendering pipeline from becoming overwhelmed by scenes that 
    vastly exceed the targeted maximum short length.

    Args:
        combined_scene_list: List of merged scenes.
        config: Settings controlling the maximum allowed scene length.

    Returns:
        A flattened list where overlong scenes have been subdivided.
    """
    result: List[Tuple[_SecondsTime, _SecondsTime]] = []
    threshold = 4 * config.max_short_length
    for scene in combined_scene_list:
        start_s = scene[0].get_seconds()
        end_s = scene[1].get_seconds()
        duration = end_s - start_s

        if duration > threshold:
            n = int(math.floor(duration / (2 * config.max_short_length)))
            if n <= 1:  # pragma: no cover
                result.append(scene)  # pragma: no cover
                continue  # pragma: no cover

            part_len = duration / n
            for i in range(n):
                part_start = start_s + i * part_len
                part_end = start_s + (i + 1) * part_len
                result.append((_SecondsTime(part_start), _SecondsTime(part_end)))
        else:
            result.append(scene)  # pragma: no cover

    return result


def find_smart_end_point(
    start_time: float,
    min_end: float,
    max_end: float,
    times: np.ndarray,
    scores: np.ndarray,
    search_window: float = 2.0,
) -> float:
    """Finds the most natural point to end a clip, avoiding mid-action cuts.

    Searches backwards from the absolute maximum end time within a specified 
    search window to find a local minimum in the action score (e.g., a moment of silence 
    or stillness), ensuring the clip doesn't cut off abruptly during an important event.

    Args:
        start_time: The locked starting time of the clip.
        min_end: The earliest allowed end time to ensure minimum clip duration.
        max_end: The absolute latest allowed end time.
        times: Array of timestamps corresponding to the score.
        scores: Array of action scores (typically audio or combined).
        search_window: How many seconds backwards from `max_end` to look for a quiet moment.

    Returns:
        The optimal ending timestamp (in seconds).
    """
    search_start = max(min_end, max_end - search_window)
    search_finish = max_end

    if search_finish <= search_start:
        return search_finish

    mask = (times >= search_start) & (times <= search_finish)
    if not np.any(mask):
        return search_finish

    t_seg = times[mask]
    s_seg = scores[mask]

    min_idx = np.argmin(s_seg)
    best_end_time = float(t_seg[min_idx])

    return best_end_time
