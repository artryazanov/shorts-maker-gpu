FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. Устанавливаем инструменты
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Устанавливаем заголовки
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 3. ИСПРАВЛЕНИЕ: Копируем драйвер в системную папку
# Файл libnvcuvid.so (который вы скопировали из WSL) мы кладем как .so.1
# чтобы программа нашла его при запуске.
COPY libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1

# Создаем ссылку .so для компилятора
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

# 4. Зависимости Python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Собираем Decord
# Обратите внимание: путь к библиотеке изменился на системный
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    make -j$(nproc) && \
    cd ../python && \
    python setup.py install && \
    cd /app && rm -rf decord

# 6. Фикс libstdc++
RUN rm /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

COPY . .

CMD ["python", "shorts.py"]