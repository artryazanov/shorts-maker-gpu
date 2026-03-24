# Installation

## 🚀 Installation via PyPI (Recommended)

Ensure you have the NVIDIA drivers and CUDA toolkit installed. Then install the package directly:

```bash
pip install shorts-maker-gpu
```

## 🛠️ Manual Setup from Source (Linux with CUDA)

Ensure you have the NVIDIA drivers and CUDA toolkit installed.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

!!! warning "GPU Availability"
    If you encounter issues with PyTorch not finding the GPU, refer to its [installation guide](https://pytorch.org/get-started/locally/) for your specific CUDA version.
