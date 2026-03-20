# Installation

## Requirements

- **NVIDIA GPU** with CUDA support.
- **NVIDIA Drivers** (compatible with CUDA 13.0+ recommended).
- Python 3.12+
- FFmpeg (used for audio extraction and NVENC encoding).
- System libraries: `libgl1`, `libglib2.0-0` (often needed for vision libraries).

Python dependencies:
- `torch`, `torchaudio` (with CUDA support)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## Manual Setup (Linux with CUDA)

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

## Docker (Recommended)

The easiest way to run this application is using Docker with the NVIDIA Container Toolkit.

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host.

Build and run:

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
