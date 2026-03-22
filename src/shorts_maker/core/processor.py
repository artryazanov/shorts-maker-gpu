import gc
import logging
import math
import random
from pathlib import Path

import PyNvCodec as nvc
import torch

from shorts_maker.analysis.audio import compute_audio_action_profile
from shorts_maker.analysis.video import compute_video_action_profile
from shorts_maker.config import ProcessingConfig
from shorts_maker.io.render import get_render_params, render_video_gpu_isolated
from shorts_maker.utils.scenes import (
    best_action_window_start,
    combine_scenes,
    detect_video_scenes_gpu,
    find_smart_end_point,
    scene_action_score,
    split_overlong_scenes,
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Core processor for generating viral shorts from long-form video.
    
    This class orchestrates the entire hardware-accelerated pipeline: scene detection,
    audio/video action profiling, intelligent clipping (smart cuts), and GPU-based 
    compositing and rendering via NVENC.

    Attributes:
        config (ProcessingConfig): Configuration settings for the generation pipeline.
    """

    def __init__(self, config: ProcessingConfig):
        """Initializes the video processor with the given configuration.

        Args:
            config: A ProcessingConfig object containing target aspect ratios, 
                scene limits, and duration constraints.
        """
        self.config = config

    def process_video(self, video_file: Path, output_dir: Path) -> None:
        """Processes a single video file to generate multiple short clips.

        Analyzes the video to find high-action scenes using combined audio-visual 
        scoring, groups them by length, determines optimal start/end points using a 
        smart cut algorithm, and dispatches the rendering process.

        Args:
            video_file: Path to the source gameplay video file.
            output_dir: Directory where the generated short clips will be saved.

        Raises:
            RuntimeError: If the rendering process fails or FFmpeg encounters an error.
        """
        logger.info("\nProcess: %s", video_file.name)

        logger.info("Detecting scenes (GPU)...")
        scene_list = detect_video_scenes_gpu(video_file)

        # Explicitly clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # pragma: no cover

        logger.info("Computing audio action profile (GPU)...")
        audio_times, audio_score = compute_audio_action_profile(video_file)

        # Explicitly clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # pragma: no cover

        logger.info("Computing video action profile (GPU)...")
        video_times, video_score = compute_video_action_profile(
            video_file,
            fps=4,
            downscale_factor=6,
        )

        # Explicitly clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # pragma: no cover

        # Pre-calculate video duration for boundary checks
        try:
            dmx = nvc.PyFFmpegDemuxer(str(video_file))
            video_duration = float(dmx.Numframes() / dmx.Framerate())
            del dmx
        except Exception:  # pragma: no cover
            logger.warning("PyNvCodec probe failed, fallback to 0 duration.")  # pragma: no cover
            video_duration = 0.0  # pragma: no cover

        processed_scene_list = combine_scenes(scene_list, self.config)
        processed_scene_list = split_overlong_scenes(processed_scene_list, self.config)

        logger.info("Scenes list with action scores:")
        for i, scene in enumerate(processed_scene_list, start=1):
            duration = scene[1].get_seconds() - scene[0].get_seconds()
            score_val = scene_action_score(
                scene, audio_times, audio_score, video_times, video_score
            )
            logger.info(
                "Scene %2d: Duration %5.1f s, ActionScore %7.3f, Start %s / Frame %d, End %s / Frame %d",
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
            key=lambda s: scene_action_score(
                s, audio_times, audio_score, video_times, video_score
            ),
            reverse=True,
        )

        logger.info("Sorted scenes list (by action score):")
        for i, scene in enumerate(sorted_processed_scene_list, start=1):
            duration = scene[1].get_seconds() - scene[0].get_seconds()
            score_val = scene_action_score(
                scene, audio_times, audio_score, video_times, video_score
            )
            logger.info(
                "Scene %2d: ActionScore %7.3f, Duration %5.1f s, Start %s / Frame %d, End %s / Frame %d",
                i,
                score_val,
                duration,
                scene[0].get_timecode(),
                scene[0].get_frames(),
                scene[1].get_timecode(),
                scene[1].get_frames(),
            )

        truncated_list = sorted_processed_scene_list[: self.config.scene_limit]

        if truncated_list:
            for i, scene in enumerate(truncated_list):
                scene_start = scene[0].get_seconds()
                scene_end = scene[1].get_seconds()
                scene_duration = scene_end - scene_start

                # STRATEGY 1: If scene fits entirely - take it all.
                # We add a small padding (1.5s) to capture the "end scene animation/fade".
                if scene_duration <= self.config.max_short_length:
                    final_start = scene_start
                    padding = 1.5
                    final_end = min(scene_end + padding, video_duration)

                    # Check if padding pushes us over max limit
                    if (final_end - final_start) > self.config.max_short_length:
                        final_end = final_start + self.config.max_short_length  # pragma: no cover

                    final_duration = final_end - final_start
                    logger.info(
                        f"Scene {i}: Full scene + padding ({final_duration:.2f}s)"
                    )

                # STRATEGY 2: Scene too long, cut best window with smart end.
                else:
                    target_duration = float(self.config.max_short_length)

                    best_start = best_action_window_start(
                        scene,
                        target_duration,
                        audio_times,
                        audio_score,
                        video_times,
                        video_score,
                    )

                    absolute_min_end = best_start + self.config.min_short_length
                    absolute_max_end = min(
                        scene_end, best_start + self.config.max_short_length
                    )

                    final_end = find_smart_end_point(
                        best_start,
                        absolute_min_end,
                        absolute_max_end,
                        audio_times,
                        audio_score,
                        search_window=5.0,
                    )

                    final_start = best_start
                    final_duration = final_end - final_start
                    logger.info(
                        f"Scene {i}: Smart Cut. Start {final_start:.2f}, End {final_end:.2f} (Duration {final_duration:.2f}s)"
                    )

                render_file_name = f"{video_file.stem} scene-{i}{video_file.suffix}"
                render_path = output_dir / render_file_name

                # Prepare render params
                params = get_render_params(
                    video_file, final_start, final_duration, self.config
                )

                # Execute GPU render
                render_video_gpu_isolated(
                    params,
                    render_path,
                    max_error_depth=self.config.max_error_depth,
                    save_ffmpeg_logs=self.config.save_ffmpeg_logs,
                )
        else:
            # No scenes found, fallback to random clip
            short_length = random.randint(
                self.config.min_short_length, self.config.max_short_length
            )

            if video_duration < self.config.max_short_length:
                adapted_short_length = min(math.floor(video_duration), short_length)
            else:
                adapted_short_length = short_length  # pragma: no cover

            min_start_point = min(
                10, math.floor(video_duration) - adapted_short_length
            )
            max_start_point = math.floor(video_duration - adapted_short_length)

            start_point = float(
                random.randint(int(min_start_point), int(max_start_point))
            )

            params = get_render_params(
                video_file,
                start_point,
                float(adapted_short_length),
                self.config,
            )

            render_video_gpu_isolated(
                params,
                output_dir / video_file.name,
                max_error_depth=self.config.max_error_depth,
                save_ffmpeg_logs=self.config.save_ffmpeg_logs,
            )
