FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Update Conda and install recent FFmpeg
RUN conda clean -i && conda install -y -c conda-forge ffmpeg

# 3. Set up environment
ENV FFMPEG_BINARY=/opt/conda/bin/ffmpeg
ENV PKG_CONFIG_PATH=/opt/conda/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/conda/lib:$LD_LIBRARY_PATH

# Ensure NVIDIA driver capabilities include video for codecs
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 4. Install codec headers
# Important: for the older ffmpeg it's better to use a pinned header version, but git master should also work
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 5. Install NVIDIA driver libraries for linking
# We install "headless" driver libraries so we have the .so files for linking.
# At runtime, the NVIDIA Container Toolkit will mount the host driver's files over these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-decode-535 \
    libnvidia-encode-535 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks, so CMake and Python can find the libraries (packages typically provide .so.1/.so.535)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

# 6. Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel scikit-build ninja && \
    pip install --no-cache-dir -r requirements.txt

# 7. Build VideoProcessingFramework (PyNvCodec)
# Required for modern hardware-accelerated video decoding.
# Set paths to FFMPEG since we installed it via conda.
RUN git clone https://github.com/NVIDIA/VideoProcessingFramework.git && \
    cd VideoProcessingFramework && \
    mkdir build && cd build && \
    cmake .. \
      -DFFMPEG_DIR:PATH="/opt/conda" \
      -DVIDEO_CODEC_SDK_DIR:PATH="/app/nv-codec-headers" \
      -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
      -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.10.so \
      -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.10 \
      -DCMAKE_INSTALL_PREFIX:PATH="/usr/local" && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    pip install . --no-build-isolation && \
    pip install src/PytorchNvCodec --no-build-isolation && \
    cd /app && rm -rf VideoProcessingFramework

# 8. C++ library fix
RUN rm /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# 9. Cleanup build-time dependencies
# Remove the installed NVIDIA libraries so the container uses the host mounted ones at runtime
RUN apt-get purge -y libnvidia-decode-535 libnvidia-encode-535 && \
    rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "shorts.py"]