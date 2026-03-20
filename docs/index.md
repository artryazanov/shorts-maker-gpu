# Shorts Maker GPU 🚀

[![PyPI version](https://img.shields.io/pypi/v/shorts-maker-gpu.svg)](https://pypi.org/project/shorts-maker-gpu/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

**Shorts Maker GPU** is a high-performance library for automatically generating vertical videos (shorts) from long gameplay recordings.

Unlike standard CPU-based solutions, this library is heavily hardware-accelerated:
* **Zero-copy architecture:** Decoding, resizing, and color space conversion happen directly on the GPU via the *NVIDIA Video Processing Framework (VPF)*.
* **PyTorch Integration:** Frame processing, background blurring, and compositing are performed using native tensor operations without transferring data back to system memory.
* **Smart Detection:** Intelligent scene recognition and "action" scoring based on both audio and video streams.

[Installation](guide/installation.md){ .md-button .md-button--primary }
[Quickstart](guide/quickstart.md){ .md-button }
