

> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (GPU Optimized)

Shorts Maker generates vertical video clips from longer gameplay footage. This Python library and CLI tool detects scenes, computes audio and video action profiles (sound intensity + visual motion), and combines them to rank scenes by overall intensity. It then crops to the desired aspect ratio and renders ready‑to‑upload shorts.

**This version has been heavily optimized for NVIDIA GPUs using CUDA.**

For the original CPU-only version, please visit [Shorts Maker](https://github.com/artryazanov/shorts-maker).

[![PyPI](https://img.shields.io/pypi/v/shorts-maker-gpu.svg)](https://pypi.org/project/shorts-maker-gpu/)
[![Downloads](https://static.pepy.tech/badge/shorts-maker-gpu)](https://pepy.tech/project/shorts-maker-gpu)
[![Tests](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml)
[![Linting](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml)
[![codecov](https://codecov.io/gh/artryazanov/shorts-maker-gpu/graph/badge.svg)](https://codecov.io/gh/artryazanov/shorts-maker-gpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

### [Read the Full Documentation 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ Features

- **GPU-Accelerated Processing**:
  - **Hardware Decoding & Resizing**: Native NVIDIA Video Processing Framework (VPF) integration via `PyNvCodec`. Decodes, resizes, and converts color spaces directly on NVDEC.
  - **Scene Detection**: Custom implementation using VPF and OpenCV.
  - **Audio Analysis**: Uses `torchaudio` on GPU for fast RMS and spectral flux calculation.
  - **Video Analysis**: Zero-copy GPU memory streaming for stable motion estimation (replaces heavy frame indices).
  - **Image Processing**: Native PyTorch operators used for heavy operations like blurring backgrounds (separable convolutions).
  - **Rendering**: Custom PyTorch+NVENC engine for high-performance rendering (MoviePy removed from render path).
  - **Robust Batch Processing**: Video processing runs in fully isolated subprocesses, completely clearing CUDA contexts between files to prevent VRAM fragmentation and OOM crashes (especially in Docker/WSL).
  - **Accurate VFR Handling**: Extracts true Presentation Timestamps (PTS) directly from video packets to prevent audio/video desync, handling Variable Frame Rate (VFR) gameplay seamlessly.
- Audio + video action scoring:
  - Combined ranking with tunable weights (defaults: audio 0.6, video 0.4).
- Scenes ranked by combined action score rather than duration.
- **Smart Scene Cutting**:
  - Preferentially selects complete scenes if they fit within the time limit.
  - **Scene Padding**: Adds a 1.5-second buffer to the end of scenes to capture exit animations and fades.
  - **Smart Trimming**: For long scenes, searches for "quiet" moments (low audio/motion) to cut, avoiding abrupt endings.
- Smart cropping with optional blurred background for non‑vertical footage.
- Retry logic during rendering to avoid spurious failures.
- Configuration via `.env` environment variables.

## 📋 Requirements

- **NVIDIA GPU** with CUDA support.
- **NVIDIA Drivers** (compatible with CUDA 13.0+ recommended).
- Python 3.12+
- FFmpeg (used for audio extraction and NVENC encoding).
- System libraries: `libgl1`, `libglib2.0-0` (often needed for vision libraries).

Python dependencies (see `pyproject.toml`):
- `torch`, `torchaudio` (with CUDA support)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Installation

### Via PyPI (Recommended)

Ensure you have the NVIDIA drivers and CUDA toolkit installed. Then install the package directly:

```bash
pip install shorts-maker-gpu
```

### Manual Setup from Source (Linux with CUDA)

Ensure you have the NVIDIA drivers and CUDA toolkit installed.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

If you encounter issues with PyTorch not finding the GPU, refer to its installation guide for your specific CUDA version.

## 💡 Usage

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

## 🐳 Docker (Recommended)

The easiest way to run this application is using Docker with the NVIDIA Container Toolkit.

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host.

Build and run:

*(Note: If the build crashes with a "Segmentation fault" or memory error, limit the CPU cores by using `docker build --cpuset-cpus="0,1" -t shorts-maker .` instead).*

```bash
docker build -t shorts-maker .

# Run with GPU access
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

Note the `--gpus all` flag, which is essential for the application to access hardware acceleration.

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Supported variables (defaults shown):
- `TARGET_RATIO_W=9` — Width part of the target aspect ratio (e.g., 9 for 9:16).
- `TARGET_RATIO_H=16` — Height part of the target aspect ratio (e.g., 16 for 9:16).
- `SCENE_LIMIT=4` — Maximum number of top scenes rendered per source video.
- `SCENE_THRESHOLD=45.0` — Threshold for scene detection cuts.
- `SKIP_FIRST_SECONDS=0.0` — Seconds to skip from the beginning of the video during scene detection.
- `X_CENTER=0.5` — Horizontal crop center in range [0.0, 1.0].
- `Y_CENTER=0.5` — Vertical crop center in range [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Maximum retry depth if rendering fails.
- `MIN_SHORT_LENGTH=15` — Minimum short length in seconds.
- `MAX_SHORT_LENGTH=179` — Maximum short length in seconds.
- `MAX_COMBINED_SCENE_LENGTH=300` — Maximum combined length (in seconds).
- `SAVE_FFMPEG_LOGS=False` — Whether to save FFmpeg logs during rendering.
- `LOG_LEVEL=WARNING` — Logging level (e.g., INFO, DEBUG, WARNING).

## 🛠️ Development

### Linting

This project uses `ruff` for fast linting.

```bash
pip install ruff
ruff check .
```

## 🧪 Running Tests

Unit tests live in the `tests/` folder. Run them with:

```bash
pytest -q
```

Note: The tests are designed to mock GPU availability if it is missing, so they can run in standard CI environments.

## 🚑 Troubleshooting

- **"internal compiler error: Segmentation fault" during `docker build`**: This typically occurs due to an Out-Of-Memory (OOM) error when Docker attempts to compile heavy C++/CUDA libraries (like VPF) using all available CPU cores. To fix this, limit the number of CPU cores used during the build process:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternatively, you can increase the RAM limit for Docker/WSL2 in your system settings).*
- **"WSL integration with distro unexpectedly stopped" / OOM during `docker run`**: Processing high-resolution video can consume significant RAM/VRAM, causing the WSL2 virtual machine to crash due to an Out-Of-Memory (OOM) error. To fix this, limit the number of CPU cores the container can use during execution by adding the `--cpus` flag:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: Ensure you are running inside the Docker container with `--gpus all` or have the correct CUDA toolkit installed locally.
- **NVENC Error**: If `h264_nvenc` fails, the script attempts to fall back to software encoding (`libx264`). Check if your GPU supports NVENC and if the drivers are up to date.

## 📄 License

This project is released under the [MIT License](LICENSE).
