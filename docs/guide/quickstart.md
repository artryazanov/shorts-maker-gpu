# Quickstart

## Usage

1. Place source videos inside the `gameplay/` directory.
2. Run the CLI tool:

```bash
shorts-maker process
```

You can optionally customize the input and output directories and scene limits:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Generated clips are written to the `generated/` directory.

During processing, the log shows an action score for each combined scene and the final list sorted by that score. The top scenes (by action intensity) are rendered first using NVENC.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Supported variables (defaults shown):
- `TARGET_RATIO_W=9` — Width part of the target aspect ratio (e.g., 9 for 9:16).
- `TARGET_RATIO_H=16` — Height part of the target aspect ratio (e.g., 16 for 9:16).
- `SCENE_LIMIT=4` — Maximum number of top scenes rendered per source video.
- `X_CENTER=0.5` — Horizontal crop center in range [0.0, 1.0].
- `Y_CENTER=0.5` — Vertical crop center in range [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Maximum retry depth if rendering fails.
- `MIN_SHORT_LENGTH=15` — Minimum short length in seconds.
- `MAX_SHORT_LENGTH=179` — Maximum short length in seconds.
- `MAX_COMBINED_SCENE_LENGTH=300` — Maximum combined length (in seconds).
