import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from moviepy import ColorClip

# Ensure the project root is on the import path.
import sys
import types
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Stub scenedetect to avoid heavy OpenCV dependency during import.
scenedetect_stub = types.ModuleType("scenedetect")
scenedetect_stub.SceneManager = object  # type: ignore
scenedetect_stub.open_video = lambda *_args, **_kwargs: None  # type: ignore

detectors_stub = types.ModuleType("scenedetect.detectors")
detectors_stub.ContentDetector = object  # type: ignore

sys.modules.setdefault("scenedetect", scenedetect_stub)
sys.modules.setdefault("scenedetect.detectors", detectors_stub)

import shorts
from shorts import (
    blur,
    combine_scenes,
    crop_clip,
    select_background_resolution,
    ProcessingConfig,
    render_video,
    scene_action_score,
    best_action_window_start,
    compute_audio_action_profile,
    compute_video_action_profile,
)


class MockTime:
    """Simple stand-in for scenedetect's time objects."""

    def __init__(self, seconds: float):
        self._seconds = seconds

    def get_seconds(self) -> float:
        return self._seconds

    # The functions below are unused in logic but required by combine_scenes
    def get_timecode(self) -> str:
        return str(self._seconds)

    def get_frames(self) -> int:
        return int(self._seconds * 30)


def make_scene(start: float, end: float):
    return (MockTime(start), MockTime(end))


def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2100) == (2160, 3840)


def test_crop_clip_to_square():
    clip = ColorClip(size=(1920, 1080), color=(255, 0, 0), duration=1)
    cropped = crop_clip(clip, 1, 1, 0.5, 0.5)
    assert cropped.size == (1080, 1080)


def test_blur_changes_image():
    image = np.zeros((10, 10))
    image[5, 5] = 1.0
    blurred = blur(image)
    assert blurred.shape == image.shape
    assert blurred[5, 5] != image[5, 5]


def test_combine_scenes_merges_short_scenes():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 5),
        make_scene(5, 7),
        make_scene(7, 9),
        make_scene(9, 11),
        make_scene(11, 13),
        make_scene(13, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 1
    start, end = combined[0]
    assert start.get_seconds() == 5
    assert end.get_seconds() == 13


def test_render_video_retries(tmp_path):
    clip = MagicMock()
    clip.fps = 30
    clip.write_videofile.side_effect = [Exception("boom"), None]
    render_video(clip, Path("out.mp4"), tmp_path, max_error_depth=1)
    assert clip.write_videofile.call_count == 2




def test_render_video_raises_after_retries(tmp_path):
    clip = MagicMock()
    clip.fps = 60
    clip.write_videofile.side_effect = Exception("fail")

    with pytest.raises(Exception):
        render_video(clip, Path("out.mp4"), tmp_path, max_error_depth=0)



def test_scene_action_score_sum():
    times = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    score = np.array([0, 10, 10, 10, 0, 0, 0], dtype=float)
    scene = make_scene(1.0, 4.0)
    total = scene_action_score(scene, times, score)
    assert total == pytest.approx(30.0, rel=1e-6)


def test_scene_action_score_empty_segment():
    times = np.array([0.0, 1.0], dtype=float)
    score = np.array([1.0, 1.0], dtype=float)
    scene = make_scene(2.0, 3.0)
    assert scene_action_score(scene, times, score) == 0.0


def test_scene_action_score_invalid_range():
    times = np.array([0.0], dtype=float)
    score = np.array([0.0], dtype=float)
    scene = make_scene(5.0, 5.0)
    assert scene_action_score(scene, times, score) == 0.0


def test_compute_audio_action_profile_stubbed(monkeypatch):
    class LibrosaStub:
        def load(self, path, sr=None, mono=True):
            return np.zeros(1000, dtype=float), 100

        class feature:
            @staticmethod
            def rms(y, frame_length=2048, hop_length=512):
                # Return shape (1, n_frames)
                return np.array([[1.0, 2.0, 3.0]], dtype=float)

        @staticmethod
        def stft(y, n_fft=2048, hop_length=512):
            # Shape (freq_bins, n_frames)
            return np.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=float,
            )

        @staticmethod
        def frames_to_time(frames, sr=100, hop_length=512):
            return np.asarray(frames, dtype=float) * 0.01

    # Patch the librosa module used inside shorts
    monkeypatch.setattr(shorts, "librosa", LibrosaStub(), raising=False)

    times, score = compute_audio_action_profile(Path("dummy.mp4"))

    assert isinstance(times, np.ndarray)
    assert isinstance(score, np.ndarray)
    assert len(times) == len(score) == 3
    # Combined score should not be constant given our stub inputs
    assert score.std() > 0



def test_best_action_window_start_picks_max_window():
    # times every 1s from 0..19
    times = np.arange(0.0, 20.0, 1.0, dtype=float)
    score = np.zeros_like(times)
    # Low action at 2..4
    score[2:5] = 1.0
    # High action at 8..10 — the best 3s window should start at 8
    score[8:11] = 2.0

    scene = make_scene(0.0, 15.0)
    start = best_action_window_start(scene, 3.0, times, score)
    assert start == pytest.approx(8.0, rel=1e-9)


def test_best_action_window_start_clamps_to_fit():
    # Increasing scores push the best window to the end, but it must clamp to fit
    times = np.arange(0.0, 6.0, 1.0, dtype=float)
    score = np.arange(len(times), dtype=float)  # 0,1,2,3,4,5

    scene = make_scene(0.0, 5.0)
    # Window 4s can only start in [0, 1]; the raw best start would be 2 -> clamp to 1
    start = best_action_window_start(scene, 4.0, times, score)
    assert start == pytest.approx(1.0, rel=1e-9)


def test_best_action_window_start_fallback_no_frames():
    times = np.arange(100.0, 110.0, 1.0, dtype=float)
    score = np.ones_like(times)
    scene = make_scene(0.0, 5.0)
    start = best_action_window_start(scene, 3.0, times, score)
    assert start == pytest.approx(0.0, rel=1e-9)


def test_best_action_window_start_short_scene():
    times = np.arange(0.0, 50.0, 1.0, dtype=float)
    score = np.ones_like(times)
    scene = make_scene(10.0, 12.0)  # duration 2s
    start = best_action_window_start(scene, 5.0, times, score)
    assert start == pytest.approx(10.0, rel=1e-9)



def test_combine_scenes_merges_interior_short_run():
    # Interior run of short scenes (< min_short_length) should be merged with neighbours,
    # not dropped. Boundary runs use middle_short_length threshold.
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=300)
    scenes = [
        make_scene(0, 8),   # long boundary run (>= middle_short_length -> kept)
        make_scene(8, 9),   # short
        make_scene(9, 10),  # short (interior run total = 2 < min -> merge)
        make_scene(10, 20), # long
    ]

    combined = combine_scenes(scenes, config)
    assert len(combined) == 2
    (s1, e1), (s2, e2) = combined
    assert s1.get_seconds() == 0 and e1.get_seconds() == 8
    # The interior short run should be merged into the next long run
    assert s2.get_seconds() == 8 and e2.get_seconds() == 20


def test_combine_scenes_splits_long_small_run_by_cap():
    # A long sequence of short scenes must be split by max_combined_scene_length, and
    # the split occurs on the previous scene boundary to avoid overlap.
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=10)

    # 20 consecutive 1-second scenes (all "short") from 0..20
    scenes = [make_scene(t, t + 1) for t in range(0, 20)]

    combined = combine_scenes(scenes, config)

    # Expect two chunks: the first is flushed when the accumulated duration reaches the cap,
    # closing at the previous boundary (end at 10), then the remainder up to the last full
    # boundary before exceeding the cap (ends at 19). The final 1s tail is dropped as a
    # boundary shorter than middle_short_length.
    assert len(combined) == 2
    (s1, e1), (s2, e2) = combined
    assert s1.get_seconds() == 0 and e1.get_seconds() == 10
    assert s2.get_seconds() == 10 and e2.get_seconds() == 19



def test_compute_video_action_profile_stubbed_basic(monkeypatch):
    # Stub VideoFileClip.iter_frames to avoid real decoding
    class VideoStub:
        def __init__(self, *_args, **_kwargs):
            self.duration = 4.0
            self.fps = 30  # source fps
        def iter_frames(self, fps=2.0, dtype="uint8", logger=None):
            # Yield exactly int(duration*fps) frames
            n = int(self.duration * fps)
            for i in range(n):
                # Toggle brightness every frame to induce motion
                val = 255 if (i % 2) else 0
                yield np.full((4, 4, 3), val, dtype=np.uint8)
        def close(self):
            pass

    monkeypatch.setattr(shorts, "VideoFileClip", VideoStub, raising=False)

    # Use a low fps so the array is small and deterministic
    times, score = compute_video_action_profile(Path("dummy.mp4"), fps=2)

    assert isinstance(times, np.ndarray)
    assert isinstance(score, np.ndarray)
    # duration 4s at 2 fps -> 8 samples
    assert len(times) == int(4.0 * 2) and len(score) == int(4.0 * 2)
    # Score should have some variance due to alternating diffs
    assert score.std() > 0


def test_compute_video_action_profile_zero_duration(monkeypatch):
    class ZeroDurStub:
        def __init__(self, *_args, **_kwargs):
            self.duration = 0.0
        def get_frame(self, t: float):  # pragma: no cover - should not be called
            raise AssertionError("get_frame should not be called for zero duration")
        def close(self):
            pass

    monkeypatch.setattr(shorts, "VideoFileClip", ZeroDurStub, raising=False)

    times, score = compute_video_action_profile(Path("dummy.mp4"), fps=5)
    assert times.size == 0 and score.size == 0


def test_scene_action_score_combines_audio_video():
    # Simple 0..4s with unit audio everywhere and a video spike at t=2
    audio_times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    audio_score = np.ones_like(audio_times)
    video_times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    video_score = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    scene = make_scene(0.0, 4.0)
    total = scene_action_score(
        scene,
        audio_times,
        audio_score,
        video_times,
        video_score,
        w_audio=0.6,
        w_video=0.4,
    )
    # audio sum = 4, video sum = 1 -> total = 0.6*4 + 0.4*1 = 2.8
    assert total == pytest.approx(2.8, rel=1e-9)


def test_best_action_window_start_prefers_video_when_weighted():
    # Audio has zero action; video has a 2s high-action segment at ~6..8
    audio_times = np.arange(0.0, 10.0, 1.0, dtype=float)
    audio_score = np.zeros_like(audio_times)

    video_times = np.arange(0.0, 10.0, 0.5, dtype=float)
    video_score = np.zeros_like(video_times)
    # Make 6.0 <= t < 8.0 high action
    video_score[(video_times >= 6.0) & (video_times < 8.0)] = 10.0

    scene = make_scene(0.0, 10.0)
    start = best_action_window_start(
        scene,
        2.0,
        audio_times,
        audio_score,
        video_times,
        video_score,
        w_audio=0.0,
        w_video=1.0,
    )
    assert start == pytest.approx(6.0, rel=1e-6)


def test_best_action_window_start_fallback_video_only():
    # Audio has no samples inside scene; video exists and should be used
    audio_times = np.array([100.0, 101.0], dtype=float)
    audio_score = np.array([1.0, 1.0], dtype=float)

    video_times = np.arange(0.0, 6.0, 1.0, dtype=float)
    video_score = np.zeros_like(video_times)
    video_score[2:5] = 2.0  # best 2s window should start at 2.0

    scene = make_scene(0.0, 5.0)
    start = best_action_window_start(
        scene,
        2.0,
        audio_times,
        audio_score,
        video_times,
        video_score,
    )
    assert start == pytest.approx(2.0, rel=1e-9)
