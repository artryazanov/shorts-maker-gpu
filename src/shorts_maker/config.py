"""Configuration definitions and loaded settings for the Shorts Maker pipeline."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseSettings):
    """Configuration values used throughout the processing pipeline.
    
    This Pydantic settings model manages all parameters required for video processing,
    including crop aspect ratios, scene generation limits, and timing boundaries for
    the generated short clips. Values can be loaded from environment variables or a `.env` file.

    Attributes:
        target_ratio_w (int): Target width for the final crop aspect ratio.
        target_ratio_h (int): Target height for the final crop aspect ratio.
        scene_limit (int): Maximum number of top scenes to render from a single video.
        x_center (float): Horizontal center point for the crop (0.0=left, 1.0=right).
        y_center (float): Vertical center point for the crop (0.0=top, 1.0=bottom).
        max_error_depth (int): Maximum number of retries if GPU rendering fails.
        min_short_length (int): Minimum permissible duration (in seconds) for a generated short.
        max_short_length (int): Maximum permissible duration (in seconds) for a generated short.
        max_combined_scene_length (int): Maximum permitted length for a contiguous block of action.
    """

    target_ratio_w: int = Field(default=1, description="Target aspect ratio width")
    target_ratio_h: int = Field(default=1, description="Target aspect ratio height")
    scene_limit: int = Field(default=6, description="Maximum scenes to render")
    x_center: float = Field(default=0.5, description="Horizontal center position (0.0 to 1.0)")
    y_center: float = Field(default=0.5, description="Vertical center position (0.0 to 1.0)")
    max_error_depth: int = Field(default=3, description="Maximum error depth for rendering")
    min_short_length: int = Field(default=15, description="Minimum short length in seconds")
    max_short_length: int = Field(default=179, description="Maximum short length in seconds")
    max_combined_scene_length: int = Field(default=300, description="Max allowed combined scene length")
    save_ffmpeg_logs: bool = Field(default=False, description="Whether to save FFmpeg logs during rendering")
    log_level: str = Field(default="WARNING", description="Logging level (e.g., INFO, DEBUG, WARNING)")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def middle_short_length(self) -> float:
        """Return the mid point between min and max short lengths."""
        return (self.min_short_length + self.max_short_length) / 2
