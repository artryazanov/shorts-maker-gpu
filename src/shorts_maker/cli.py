"""Command Line Interface for Shorts Maker, orchestrating the entire processing pipeline."""

import logging
import multiprocessing
from pathlib import Path

import typer
from rich.logging import RichHandler

from shorts_maker.config import ProcessingConfig
from shorts_maker.core.processor import VideoProcessor

app = typer.Typer(help="GPU-accelerated shorts generator.")

def _process_video_worker(config: ProcessingConfig, video_file: Path, output_dir: Path) -> None:
    """Isolated worker for processing a single video.

    Ensures that PyTorch and VPF memory is entirely cleared upon exit.
    """
    logging.basicConfig(
        level=config.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    processor = VideoProcessor(config)
    processor.process_video(video_file, output_dir)


@app.command()
def process(
    input_dir: Path = typer.Option(
        Path("gameplay"), help="Directory with source videos"
    ),
    output_dir: Path = typer.Option(Path("generated"), help="Output directory"),
    scene_limit: int = typer.Option(None, help="Override scene limit from config"),
) -> None:
    """Processes long gameplay videos to generate hardware-accelerated viral shorts.

    This command orchestrates the entire pipeline: reading source files, analyzing audio
    and video streams for high-action scenes, evaluating smart crop boundaries, and finally 
    rendering the optimal shorts using GPU acceleration (NVENC).

    Args:
        input_dir: Directory containing the source long-form videos.
        output_dir: Directory where generated short clips will be saved.
        scene_limit: Optional integer to override the maximum amount of clips generated 
            per standard video (defaults to the amount in .env config).

    Raises:
        typer.Exit: If the provided `input_dir` does not exist or is not a valid directory.
    """
    config = ProcessingConfig()

    # Configure logging for CLI
    logging.basicConfig(
        level=config.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger = logging.getLogger("shorts_maker")
    if scene_limit:
        config.scene_limit = scene_limit

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.warning(f"No '{input_dir}' directory found. Exiting.")
        raise typer.Exit(code=1)

    for video_file in input_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in [
            ".mp4",
            ".mkv",
            ".mov",
        ]:
            logger.info(f"\n--- Spawning isolated process for: {video_file.name} ---")
            
            # Using 'spawn' guarantees a clean process without inherited CUDA contexts
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(
                target=_process_video_worker, 
                args=(config, video_file, output_dir)
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                logger.error(f"Processing failed for {video_file.name} with exit code {p.exitcode}")  # pragma: no cover
                if p.exitcode in (-9, 137):  # pragma: no cover
                    logger.error("Process was likely OOM killed by Docker/WSL.")  # pragma: no cover


if __name__ == "__main__":
    app()  # pragma: no cover
