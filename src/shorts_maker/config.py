from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseSettings):
    """Configuration values used throughout the processing pipeline."""

    target_ratio_w: int = Field(default=1, description="Target aspect ratio width")
    target_ratio_h: int = Field(default=1, description="Target aspect ratio height")
    scene_limit: int = Field(default=6, description="Maximum scenes to render")
    x_center: float = Field(default=0.5, description="Horizontal center position (0.0 to 1.0)")
    y_center: float = Field(default=0.5, description="Vertical center position (0.0 to 1.0)")
    max_error_depth: int = Field(default=3, description="Maximum error depth for rendering")
    min_short_length: int = Field(default=15, description="Minimum short length in seconds")
    max_short_length: int = Field(default=179, description="Maximum short length in seconds")
    max_combined_scene_length: int = Field(default=300, description="Max allowed combined scene length")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def middle_short_length(self) -> float:
        """Return the mid point between min and max short lengths."""
        return (self.min_short_length + self.max_short_length) / 2
