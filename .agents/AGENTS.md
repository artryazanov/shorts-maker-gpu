# Shorts Maker (GPU Optimized) - Agent Context & Instructions

Welcome to the `shorts-maker-gpu` codebase! This document provides crucial context, architectural guidelines, and gotchas for AI coding agents interacting with this repository.

## 📌 Project Overview
**Shorts Maker** is a Python CLI tool that automatically analyzes long-form gameplay footage, detects high-intensity action scenes (via audio and video analysis), and crops them into vertical YouTube Shorts / TikToks.
This variant is **heavily optimized for NVIDIA GPUs**, completely bypassing CPU bottlenecks by using zero-copy GPU memory pipelines.

## 🛠️ Tech Stack & Dependencies
- **Language**: Python 3.12+
- **Core GPU Libraries**: 
  - `PyTorch` (CUDA) - Used for tensor manipulation, convolutions (blurring), and tensor compositing.
  - `PyNvCodec` / `PytorchNvCodec` (Video Processing Framework / VPF) - Hardware video demuxing, decoding, and color conversion.
- **Audio Processing**: `torchaudio` (CUDA)
- **Video Encoding**: `FFmpeg` subprocess piping raw bytes directly to `hevc_nvenc` / `h264_nvenc`.
- **Formatting/Linting**: `ruff`
- **Type Checking**: `mypy`
- **Testing**: `pytest`

## 🏗️ Core Architecture
The pipeline is designed to keep video frames in VRAM (GPU memory) as much as possible:

1. **Analysis & Scene Detection (`src/shorts_maker/analysis/`)**:
   - Audio is extracted and analyzed for RMS/Spectral flux (`torchaudio`).
   - Video frames are decoded natively to GPU via `GPUVideoStreamer` and subsampled. Mean absolute pixel differences calculate the video action score.
2. **Core Processor (`src/shorts_maker/core/processor.py`)**:
   - Combines audio/video scores to rank scenes.
   - Calculates smart padding and cuts to hit the required duration (e.g., 60 seconds).
3. **Streaming & Decoding (`src/shorts_maker/io/streamer.py`)**:
   - `GPUVideoStreamer` wraps VPF. It demuxes video into compressed packets, decodes them to GPU surfaces, and converts them directly to PyTorch tensors (`(N, H, W, 3)`).
4. **Rendering (`src/shorts_maker/io/render.py`)**:
   - Streams batches of GPU tensors.
   - Applies background blurring (native PyTorch separable convolutions) and foreground cropping/compositing.
   - Pipes raw RGB/YUV byte arrays to an FFmpeg subprocess for NVENC hardware encoding.
   - **Crucial**: Runs in an isolated `multiprocessing` process to guarantee VRAM cleanup and avoid CUDA OOM fragmentation.

## ⚠️ Critical Gotchas & Developer Guidelines

### 1. Presentation Timestamps (PTS) and Audio Sync
- **NEVER** use the PTS of a demuxed packet (`pkt_data.pts`) to determine the timestamp of the *subsequently* decoded frame! 
- Modern codecs (H.264/HEVC) use B-frames, meaning compressed packets are read in **Decode Order**, while decoded frames are yielded by the hardware decoder in **Display Order**.
- Using the demuxed packet PTS will cause severe non-monotonic time jitter, breaking time-based frame dropping logic (e.g., causing a 60fps video to stutter and look like 20fps).
- **Audio Desync Trap**: Conversely, you CANNOT simply reset timestamps to `0.0` or completely ignore the original PTS! Media containers (MP4, ShadowPlay recordings) often have a large initial PTS offset. If you ignore this initial offset, FFmpeg (which relies on absolute container PTS for `-ss`) will extract audio starting from the wrong absolute time, causing severe audio/video desync.
- **Rule**: To achieve perfect 60fps pacing *and* perfect audio sync, read and preserve the absolute container PTS of the very first packet (`first_packet_time`). Then, for all subsequent frames, enforce strict monotonic time based on that offset: `packet_time = first_packet_time + (frame_idx - start_frame) / fps`.

### 2. VRAM Memory Management
- GPU memory is scarce and easily fragmented.
- When manipulating large video tensor batches, always manually `del` large intermediate tensors (`frames`, `bg_frames`, `out_tensor`) at the end of loops.
- Explicitly call `gc.collect()` and `torch.cuda.empty_cache()` at significant pipeline boundaries (e.g., after scene detection, before rendering).

### 3. VPF (Video Processing Framework) Fragility
- VPF API calls (`DecodeSurfaceFromPacket`, `SurfaceConverter`, `PyFFmpegDemuxer`) are notoriously sensitive. Unhandled exceptions or incorrect surface instantiations will silently leak CUDA memory or segfault.
- Always wrap demuxing and decoding logic in `try/except` blocks or handle `Empty()` checks appropriately.

### 4. Running and Testing
- The project is typically run natively inside a Windows WSL2 Ubuntu environment or via Docker.
- Since it relies on CUDA, you must ensure the environment has `--gpus all` (Docker) or CUDA drivers (WSL2).
- When running tests or linters, prefer using `uv run`, or activating the virtual environment (`.venv/bin/python`). For WSL/Windows interoperability, prefix bash commands with `wsl -d Ubuntu-24.04 bash -c "..."` if triggered from Windows PowerShell.
- Docker builds may fail with OOM / Segfaults during VPF compilation if not constrained. (Advise the user to use `--cpuset-cpus`).

### 5. Git and Version Control
- **NEVER** run `git commit` or `git push` on behalf of the user. The user prefers to review, commit, and push all changes themselves. Make your modifications to the code and inform the user when they are ready to be committed.

## 🧹 Code Style
- Use strict type hinting (`mypy --strict`).
- Run `ruff check .` and `ruff format .` before finalizing any changes.
- Document classes and complex functions following Google Docstring format.
