# Shorts Maker GPU 🚀

[![PyPI version](https://img.shields.io/pypi/v/shorts-maker-gpu.svg)](https://pypi.org/project/shorts-maker-gpu/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/testing.yml)
[![Linting](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml/badge.svg)](https://github.com/artryazanov/shorts-maker-gpu/actions/workflows/linting.yml)
[![codecov](https://codecov.io/gh/artryazanov/shorts-maker-gpu/graph/badge.svg)](https://codecov.io/gh/artryazanov/shorts-maker-gpu)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)

**Shorts Maker GPU** is a high-performance library for automatically generating vertical videos (shorts) from long gameplay recordings.

Unlike standard CPU-based solutions, this library is heavily hardware-accelerated:
* **Zero-copy architecture:** Decoding, resizing, and color space conversion happen directly on the GPU via the *NVIDIA Video Processing Framework (VPF)*.
* **PyTorch Integration:** Frame processing, background blurring, and compositing are performed using native tensor operations without transferring data back to system memory.
* **Smart Detection:** Intelligent scene recognition and "action" scoring based on both audio and video streams.

### [Read the Full Documentation 📚](https://artryazanov.github.io/shorts-maker-gpu/)
