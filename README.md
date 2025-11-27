# Shorts Maker

Shorts Maker generates vertical video clips from longer gameplay footage. The
script detects scenes, computes audio and video action profiles (sound intensity + visual motion) and combines them to rank scenes by
overall intensity, crops to the desired aspect ratio, and renders ready‑to‑upload shorts.

## Features

- Automatic scene detection using `scenedetect`
- Audio + video action scoring:
  - Audio with `librosa` (RMS loudness + spectral flux)
  - Video motion via per-frame luma differences
  - Combined ranking with tunable weights (defaults: audio 0.6, video 0.4)
- Scenes ranked by combined action score (audio + video) rather than duration
- Smart cropping with optional blurred background for non‑vertical footage
- Retry logic during rendering to avoid spurious failures
- Configuration via `.env` environment variables (safe defaults via `ProcessingConfig`)
- Tested with `pytest`

## Requirements

- Python 3.11+
- FFmpeg (required by `moviepy`)
- See `requirements.txt` for Python dependencies (includes `librosa`, `soundfile`, `moviepy`, `scenedetect`, `scipy`)

## Installation

```bash
git clone https://github.com/artryazanov/shorts-maker.git
cd shorts-maker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If FFmpeg is not already installed on your system, refer to the
[MoviePy documentation](https://zulko.github.io/moviepy/install.html) for
installation instructions.

## Usage

1. Place source videos inside the `gameplay/` directory.
2. Run the script:

```bash
python shorts.py
```

3. Generated clips are written to the `generated/` directory.

During processing, the log shows an action score for each combined scene and the
final list sorted by that score. The top scenes (by action intensity) are
rendered first, up to `SCENE_LIMIT` per source video.

Environment variables from a `.env` file will be loaded automatically if
present.

## How it works

- Detect scenes with `scenedetect` and merge adjacent short scenes to reach a
  reasonable duration.
- Compute action profiles directly from the video file:
  - Audio profile with `librosa`: RMS loudness and spectral flux are normalized and smoothed; a per-frame score is computed as `0.6 * RMS + 0.4 * flux`.
  - Video profile with `moviepy`: frames are sampled at a fixed FPS, motion is estimated via mean absolute difference of grayscale luma between consecutive frames, then z-normalized and smoothed.
- For each scene, compute a combined action score (audio + video) and use it for logging and sorting (`scene_action_score`).
- When extracting a short from a scene, find the best start time via a sliding window over the combined profile (`best_action_window_start`).
- Sort scenes by the combined score (descending) and pick the top ones.
- Crop to the target aspect ratio; if needed, compose over a blurred background.
- Render clips with retry logic for resilience.

### Configuration

- Copy `.env.example` to `.env` and adjust values as needed.
- All variables are optional; missing or invalid values fall back to safe defaults.

Supported variables (defaults shown):
- `TARGET_RATIO_W=1` — Width part of the target aspect ratio (e.g., 9 for 9:16).
- `TARGET_RATIO_H=1` — Height part of the target aspect ratio (e.g., 16 for 9:16).
- `SCENE_LIMIT=6` — Maximum number of top scenes rendered per source video.
- `X_CENTER=0.5` — Horizontal crop center in range [0.0, 1.0].
- `Y_CENTER=0.5` — Vertical crop center in range [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Maximum retry depth if rendering fails.
- `MIN_SHORT_LENGTH=15` — Minimum short length in seconds.
- `MAX_SHORT_LENGTH=179` — Maximum short length in seconds.
- `MAX_COMBINED_SCENE_LENGTH=300` — Maximum combined length (in seconds) when merging adjacent short scenes.

Example `.env`:
```env
# Short generation defaults
TARGET_RATIO_W=1
TARGET_RATIO_H=1
SCENE_LIMIT=6
X_CENTER=0.5
Y_CENTER=0.5
MAX_ERROR_DEPTH=3
MIN_SHORT_LENGTH=15
MAX_SHORT_LENGTH=179
MAX_COMBINED_SCENE_LENGTH=300
```

## Docker

Build and run using Docker:

```bash
docker build -t shorts-maker .
docker run --rm \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

## Running Tests

Unit tests live in the `tests/` folder. Run them with:

```bash
pytest -q
```

## Troubleshooting

- FFmpeg not found: install FFmpeg and ensure it is on your PATH.

## Acknowledgments

Thank the Binary-Bytes for the original code and idea: https://github.com/Binary-Bytes/Auto-YouTube-Shorts-Maker

## License

This project is released under the [Unlicense](LICENSE).
