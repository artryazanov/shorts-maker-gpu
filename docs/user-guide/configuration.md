# Configuration

Shorts Maker GPU is configured via environment variables.

To get started, copy the example configuration file:
```bash
cp .env.example .env
```
And adjust values as needed.

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TARGET_RATIO_W` | `9`   | Width part of the target aspect ratio (for 9:16). |
| `TARGET_RATIO_H` | `16`  | Height part of the target aspect ratio (for 9:16). |
| `SCENE_LIMIT`    | `4`   | Maximum number of top scenes rendered per source video. |
| `SCENE_THRESHOLD`| `45.0`| Threshold for scene detection cuts. |
| `X_CENTER`       | `0.5` | Horizontal crop center in range `[0.0, 1.0]`. |
| `Y_CENTER`       | `0.5` | Vertical crop center in range `[0.0, 1.0]`. |
| `MAX_ERROR_DEPTH`| `3`   | Maximum retry depth if rendering fails. |
| `MIN_SHORT_LENGTH`| `15` | Minimum short length in seconds. |
| `MAX_SHORT_LENGTH`| `179`| Maximum short length in seconds. |
| `MAX_COMBINED_SCENE_LENGTH`| `300` | Maximum combined length (in seconds). |
| `SAVE_FFMPEG_LOGS` | `False` | Whether to save FFmpeg logs during rendering. |
| `LOG_LEVEL`        | `WARNING`| Logging level (e.g., INFO, DEBUG, WARNING). |
