# Usage

There are two primary ways to run Shorts Maker GPU: using Docker (which handles the complex dependencies for you) or by running it directly via the CLI.

## 🐳 Docker (Recommended)

The easiest way to run this application is using Docker with the NVIDIA Container Toolkit.

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host.

Build and run:

!!! note "Out of Memory on Build"
    If the build crashes with a "Segmentation fault" or memory error, limit the CPU cores by using `docker build --cpuset-cpus="0,1" -t shorts-maker .` instead.

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

!!! tip "Hardware Acceleration"
    Note the `--gpus all` flag, which is essential for the application to access hardware acceleration.

---

## 💡 Local CLI Usage

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

!!! info "Processing Output"
    During processing, the log shows an action score for each combined scene and the final list sorted by that score. The top scenes (by action intensity) are rendered first using NVENC.
