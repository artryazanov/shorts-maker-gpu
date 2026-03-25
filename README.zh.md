> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (GPU 优化版)

Shorts Maker 可将较长的游戏录像生成为竖屏短视频片段。这个 Python 库和 CLI 工具可以检测场景，计算音频和视频的动作特征（声音强度 + 视觉运动），并将它们结合起来，根据整体激烈程度对场景进行排名。随后，它会将其裁剪成所需的宽高比，并渲染出可直接上传的短视频 (shorts)。

**此版本已使用 CUDA 针对 NVIDIA GPU 进行了深度优化。**

如需纯 CPU 的原始版本，请访问 [Shorts Maker](https://github.com/artryazanov/shorts-maker)。

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

### [阅读完整文档 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ 特性

- **GPU 加速处理**：
  - **硬件解码与缩放**：通过 `PyNvCodec` 原生集成 NVIDIA 视频处理框架 (VPF)。直接在 NVDEC 上进行解码、缩放和色彩空间转换。
  - **场景检测**：使用 VPF 和 OpenCV 的自定义实现。
  - **音频分析**：在 GPU 上使用 `torchaudio` 进行快速 RMS 和频谱通量计算。
  - **视频分析**：零拷贝 GPU 内存流式传输，用于稳定的运动估计（替代繁重的帧索引）。
  - **图像处理**：使用原生 PyTorch 算子执行背景模糊等繁重操作（可分离卷积）。
  - **渲染**：自定义的 PyTorch+NVENC 引擎，用于高性能渲染（渲染路径中移除了 MoviePy）。
  - **稳健的批量处理**：视频处理在完全隔离的子进程中运行，在文件之间彻底清除 CUDA 上下文，防止显存碎片和 OOM 崩溃（尤其是在 Docker/WSL 中）。
- 音频 + 视频动作评分：
  - 权重可调的综合排名（默认值：音频 0.6，视频 0.4）。
- 根据综合动作评分而不是时长对场景进行排名。
- **智能场景剪辑**：
  - 如果在时间限制内，优先选择完整的场景。
  - **场景缓冲**：在场景末尾添加 1.5 秒的缓冲，以捕捉退出动画和淡出效果。
  - **智能裁剪**：对于较长的场景，会寻找“安静”时刻（低音频/低运动）进行裁剪，避免突兀的结尾。
- 智能裁剪，并为非竖屏素材提供可选的模糊背景。
- 渲染期间的重试逻辑，避免偶然性失败。
- 通过 `.env` 环境变量进行配置。

## 📋 环境要求

- 支持 CUDA 的 **NVIDIA GPU**。
- **NVIDIA 驱动**（推荐兼容 CUDA 13.0+）。
- Python 3.12+
- FFmpeg（用于音频提取和 NVENC 编码）。
- 系统库：`libgl1`，`libglib2.0-0`（视觉库通常需要）。

Python 依赖项（参见 `pyproject.toml`）：
- `torch`, `torchaudio`（支持 CUDA）
- `PyNvCodec`, `PytorchNvCodec`（视频处理框架）

## 🚀 安装

### 通过 PyPI 安装（推荐）

请确保已安装 NVIDIA 驱动和 CUDA 工具包。然后直接安装该包：

```bash
pip install shorts-maker-gpu
```

### 从源码手动设置（带有 CUDA 的 Linux）

请确保已安装 NVIDIA 驱动和 CUDA 工具包。

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# 安装库及其依赖项
pip install -e .
```

如果您遇到 PyTorch 找不到 GPU 的问题，请参阅针对您特定 CUDA 版本的安装指南。

## 💡 使用方法

1. 将源视频放置在 `gameplay/` 目录中。
2. 运行 CLI 工具：

```bash
shorts-maker process
```

您可以有选择地自定义输入和输出目录以及场景限制：
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. 生成的片段会写入 `generated/` 目录。

处理期间，日志会显示每个合并场景的动作评分以及按该评分排序的最终列表。排名靠前的场景（根据动作激烈程度）将使用 NVENC 优先渲染。

## 🐳 Docker（推荐）

运行此应用程序最简单的方法是使用 Docker 配合 NVIDIA Container Toolkit。

**前提条件**：主机上必须已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

构建并运行：

*（注意：如果构建时因“Segmentation fault”或内存错误而崩溃，请改用 `docker build --cpuset-cpus="0,1" -t shorts-maker .` 来限制 CPU 核心数）。*

```bash
docker build -t shorts-maker .

# 使用 GPU 权限运行
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

请注意 `--gpus all` 标志，这对于应用程序访问硬件加速至关重要。

## ⚙️ 配置

将 `.env.example` 复制为 `.env` 并根据需要调整各个值。

支持的变量（显示为默认值）：
- `TARGET_RATIO_W=9` — 目标宽高比的宽度部分（例如，9:16 中的 9）。
- `TARGET_RATIO_H=16` — 目标宽高比的高度部分（例如，9:16 中的 16）。
- `SCENE_LIMIT=4` — 每个源视频渲染的顶级场景的最大数量。
- `SCENE_THRESHOLD=45.0` — 场景检测切割的阈值。
- `X_CENTER=0.5` — 水平裁剪中心，范围在 [0.0, 1.0]。
- `Y_CENTER=0.5` — 垂直裁剪中心，范围在 [0.0, 1.0]。
- `MAX_ERROR_DEPTH=3` — 渲染失败时的最大重试深度。
- `MIN_SHORT_LENGTH=15` — 短视频的最小长度（秒）。
- `MAX_SHORT_LENGTH=179` — 短视频的最大长度（秒）。
- `MAX_COMBINED_SCENE_LENGTH=300` — 最大合并长度（秒）。
- `SAVE_FFMPEG_LOGS=False` — 是否在渲染期间保存 FFmpeg 日志。
- `LOG_LEVEL=WARNING` — 日志级别（例如：INFO, DEBUG, WARNING）。

## 🛠️ 开发

### 代码检查 (Linting)

本项目使用 `ruff` 进行快速代码检查。

```bash
pip install ruff
ruff check .
```

## 🧪 运行测试

单元测试位于 `tests/` 文件夹中。使用以下命令运行：

```bash
pytest -q
```

注意：测试在设计上会在缺少 GPU 时模拟其可用性，以便它们可以在标准的 CI 环境中运行。

## 🚑 故障排除

- **在 `docker build` 期间出现 "internal compiler error: Segmentation fault"**：这通常是因为 Docker 尝试使用所有可用的 CPU 核心编译繁重的 C++/CUDA 库（如 VPF）时，发生了内存不足 (OOM) 错误。如需解决此问题，请限制构建过程中使用的 CPU 核心数量：
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *（或者，您也可以在系统设置中增加 Docker/WSL2 的 RAM 限制）。*
- **在 `docker run` 期间出现 "WSL integration with distro unexpectedly stopped" / OOM**：处理高分辨率视频可能会消耗大量 RAM/VRAM，由于内存不足 (OOM) 错误导致 WSL2 虚拟机崩溃。如需解决此问题，请通过添加 `--cpus` 标志，限制容器在执行期间可使用的 CPU 核心数：
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**：请确保您在 Docker 容器内部使用 `--gpus all` 运行，或者在本地正确安装了 CUDA 工具包。
- **NVENC Error**：如果 `h264_nvenc` 失败，脚本会尝试回退到软件编码 (`libx264`)。请检查您的 GPU 是否支持 NVENC，以及驱动程序是否为最新版本。

## 📄 许可证

本项目在 [MIT 许可证](LICENSE) 下发布。