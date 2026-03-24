# 🎬 Shorts Maker (GPU Optimized)

Shorts Maker generates vertical video clips from longer gameplay footage. This Python library and CLI tool detects scenes, computes audio and video action profiles (sound intensity + visual motion), and combines them to rank scenes by overall intensity. It then crops to the desired aspect ratio and renders ready‑to‑upload shorts.

**This version has been heavily optimized for NVIDIA GPUs using CUDA.**

For the original CPU-only version, please visit [Shorts Maker](https://github.com/artryazanov/shorts-maker).

[![PyPI](https://img.shields.io/pypi/v/shorts-maker-gpu.svg)](https://pypi.org/project/shorts-maker-gpu/)
[![Tests](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml)
[![Linting](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml)
[![codecov](https://codecov.io/gh/artryazanov/shorts-maker-gpu/graph/badge.svg)](https://codecov.io/gh/artryazanov/shorts-maker-gpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

## ✨ Features

- **GPU-Accelerated Processing**:
  - **Hardware Decoding & Resizing**: Native NVIDIA Video Processing Framework (VPF) integration via `PyNvCodec`. Decodes, resizes, and converts color spaces directly on NVDEC.
  - **Scene Detection**: Custom implementation using VPF and OpenCV.
  - **Audio Analysis**: Uses `torchaudio` on GPU for fast RMS and spectral flux calculation.
  - **Video Analysis**: Zero-copy GPU memory streaming for stable motion estimation (replaces heavy frame indices).
  - **Image Processing**: Native PyTorch operators used for heavy operations like blurring backgrounds (separable convolutions).
  - **Rendering**: Custom PyTorch+NVENC engine for high-performance rendering (MoviePy removed from render path).
  - **Robust Batch Processing**: Video processing runs in fully isolated subprocesses, completely clearing CUDA contexts between files to prevent VRAM fragmentation and OOM crashes (especially in Docker/WSL).
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

## 📄 License

This project is released under the [MIT License](https://github.com/artryazanov/shorts-maker-gpu/blob/main/LICENSE).
