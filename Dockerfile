FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies (single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    wget \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Environment variables
ENV FFMPEG_BINARY=/usr/bin/ffmpeg
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 3. NVIDIA MAGIC: Allow pass-through of video libraries (NVENC/NVDEC) from the host at runtime
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# 4. Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --break-system-packages --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel scikit-build ninja && \
    pip install --break-system-packages --no-cache-dir -r requirements.txt

# 5. BUILD BLOCK: Install drivers -> Build VPF -> Remove drivers (ALL IN ONE LAYER)
# This guarantees that the hardcoded driver version (550) is used only for linking
# and DOES NOT end up in the final image.
RUN apt-get update && \
    # 5.1 Install stub libraries for linking
    apt-get install -y --no-install-recommends libnvidia-decode-550 libnvidia-encode-550 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    \
    # 5.2 Build nv-codec-headers
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
    cd /tmp/nv-codec-headers && make install && \
    \
    # 5.3 Build VideoProcessingFramework
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git /tmp/VPF && \
    cd /tmp/VPF && mkdir build && cd build && \
    cmake .. \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DFFMPEG_DIR:PATH="/usr" \
      -DVIDEO_CODEC_SDK_DIR:PATH="/tmp/nv-codec-headers" \
      -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
      -DPYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.12.so" \
      -DPYTHON_INCLUDE_DIR="/usr/include/python3.12" \
      -DCMAKE_INSTALL_PREFIX:PATH="/usr/local" && \
    make -j$(nproc) && make install && \
    cd .. && \
    export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES='80;86;89;90'" && \
    pip install --break-system-packages . --no-build-isolation && \
    pip install --break-system-packages src/PytorchNvCodec --no-build-isolation && \
    \
    # 5.4 COMPLETE CLEANUP IN THE SAME LAYER
    apt-get purge -y libnvidia-decode-550 libnvidia-encode-550 && \
    apt-get autoremove -y && \
    rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    rm -rf /var/lib/apt/lists/* /tmp/nv-codec-headers /tmp/VPF

COPY . .

CMD ["python", "shorts.py"]