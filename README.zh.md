> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (GPU 优化版)

Shorts Maker 用于从较长的游戏录像中生成垂直的短视频片段。这个 Python 库和命令行界面 (CLI) 工具可以检测场景，计算音频和视频的动作特征（声音强度 + 视觉运动），并将它们结合起来按整体强度对场景进行排名。然后，它会将其裁剪为所需的宽高比，并渲染出可直接上传的短视频 (shorts)。

**此版本已针对使用 CUDA 的 NVIDIA GPU 进行了深度优化。**

如需原始的纯 CPU 版本，请访问 [Shorts Maker](https://github.com/artryazanov/shorts-maker)。

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

- **GPU 加速处理**:
  - **硬件解码和调整大小**: 通过 `PyNvCodec` 原生集成 NVIDIA 视频处理框架 (VPF)。直接在 NVDEC 上解码、调整大小和转换色彩空间。
  - **场景检测**: 使用 VPF 和 OpenCV 的自定义实现。
  - **音频分析**: 在 GPU 上使用 `torchaudio` 进行快速 RMS 和频谱通量计算。
  - **视频分析**: 零拷贝 GPU 内存流，用于稳定的运动估计（替代繁重的帧索引）。
  - **图像处理**: 使用原生 PyTorch 算子执行模糊背景等繁重操作（可分离卷积）。
  - **渲染**: 自定义 PyTorch+NVENC 引擎，用于高性能渲染（渲染路径中已移除 MoviePy）。
  - **稳健的批处理**: 视频处理在完全隔离的子进程中运行，在文件之间完全清除 CUDA 上下文，以防止显存 (VRAM) 碎片和 OOM（内存溢出）崩溃（特别是在 Docker/WSL 中）。
- 音频 + 视频动作评分:
  - 具有可调权重的组合排名（默认值：音频 0.6，视频 0.4）。
- 场景按综合动作得分而非时长进行排名。
- **智能场景剪辑**:
  - 如果符合时间限制，优先选择完整的场景。
  - **场景填充 (Padding)**: 在场景末尾添加 1.5 秒的缓冲，以捕捉退出动画和淡出效果。
  - **智能修剪**: 对于较长的场景，搜索“安静”时刻（低音频/低运动量）进行剪辑，避免突兀的结尾。
- 智能裁剪，并为非垂直画面提供可选的模糊背景。
- 渲染期间的重试逻辑，以避免假性失败。
- 通过 `.env` 环境变量进行配置。

## 📋 环境要求

- 支持 CUDA 的 **NVIDIA GPU**。
- **NVIDIA 驱动程序**（建议兼容 CUDA 13.0+）。
- Python 3.12+
- FFmpeg（用于音频提取和 NVENC 编码）。
- 系统库：`libgl1`, `libglib2.0-0`（通常视觉库需要用到）。

Python 依赖项（见 `pyproject.toml`）：
- `torch`, `torchaudio`（支持 CUDA）
- `PyNvCodec`, `PytorchNvCodec`（视频处理框架）

## 🚀 安装

### 通过 PyPI 安装（推荐）

请确保已安装 NVIDIA 驱动程序和 CUDA 工具包。然后直接安装该包：

```bash
pip install shorts-maker-gpu
```

### 从源码手动设置（使用 CUDA 的 Linux 系统）

请确保已安装 NVIDIA 驱动程序和 CUDA 工具包。

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# 安装库及其依赖项
pip install -e .
```

如果您遇到 PyTorch 无法找到 GPU 的问题，请参阅针对您特定 CUDA 版本的 PyTorch 安装指南。

## 💡 使用方法

1. 将源视频放置在 `gameplay/` 目录中。
2. 运行 CLI 工具：

```bash
shorts-maker process
```

您可以选择自定义输入和输出目录以及场景限制：
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. 生成的片段将写入 `generated/` 目录。

在处理过程中，日志将显示每个组合场景的动作得分，并最终按该得分对列表进行排序。得分最高（动作最激烈）的场景会优先使用 NVENC 进行渲染。

## 🐳 Docker（推荐）

运行此应用程序最简单的方法是使用装有 NVIDIA Container Toolkit 的 Docker。

**先决条件**：主机上必须已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

构建并运行：

*(注意：如果构建因“段错误 (Segmentation fault)”或内存错误而崩溃，请改用 `docker build --cpuset-cpus="0,1" -t shorts-maker .` 来限制 CPU 核心数)。*

```bash
docker build -t shorts-maker .

# 允许访问 GPU 并运行
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
- `SCENE_LIMIT=4` — 每个源视频渲染的最佳场景最大数量。
- `X_CENTER=0.5` — 水平裁剪中心，范围为 [0.0, 1.0]。
- `Y_CENTER=0.5` — 垂直裁剪中心，范围为 [0.0, 1.0]。
- `MAX_ERROR_DEPTH=3` — 渲染失败时的最大重试深度。
- `MIN_SHORT_LENGTH=15` — 短视频最小长度（秒）。
- `MAX_SHORT_LENGTH=179` — 短视频最大长度（秒）。
- `MAX_COMBINED_SCENE_LENGTH=300` — 最大组合长度（秒）。
- `SAVE_FFMPEG_LOGS=False` — 是否在渲染期间保存 FFmpeg 日志。

## 🛠️ 开发

### 代码检查 (Linting)

本项目使用 `ruff` 进行快速代码检查。

```bash
pip install ruff
ruff check .
```

## 🧪 运行测试

单元测试位于 `tests/` 文件夹中。运行方式如下：

```bash
pytest -q
```

注意：测试用例被设计为在缺少 GPU 时模拟 GPU 的可用性，以便它们可以在标准的 CI 环境中运行。

## 🚑 故障排除

- **`docker build` 期间出现 "internal compiler error: Segmentation fault"**：这通常是由于 Docker 尝试使用所有可用的 CPU 核心来编译繁重的 C++/CUDA 库（如 VPF）时发生内存溢出 (OOM) 错误所致。要修复此问题，请限制构建过程中使用的 CPU 核心数量：
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(或者，您也可以在系统设置中增加 Docker/WSL2 的 RAM 限制)。*
- **"Torch not installed" / "CUDA not available"**：请确保您在 Docker 容器内部使用 `--gpus all` 运行，或者在本地正确安装了对应的 CUDA 工具包。
- **NVENC 错误**：如果 `h264_nvenc` 失败，脚本会尝试回退到软件编码 (`libx264`)。请检查您的 GPU 是否支持 NVENC 以及驱动程序是否为最新版本。

## 📄 许可证

本项目在 [MIT 许可证](LICENSE) 下发布。