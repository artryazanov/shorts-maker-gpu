> 🌐 **Idiomas:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (Optimizado para GPU)

Shorts Maker genera clips de video verticales a partir de videos de partidas (gameplays) más largos. Esta biblioteca de Python y herramienta CLI detecta escenas, calcula perfiles de acción de audio y video (intensidad del sonido + movimiento visual) y los combina para clasificar las escenas por su intensidad general. Luego, recorta al formato deseado y renderiza los "shorts" listos para ser subidos.

**Esta versión ha sido fuertemente optimizada para GPUs NVIDIA utilizando CUDA.**

Para la versión original solo para CPU, por favor visite [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

### [Lee la documentación completa 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ Características

- **Procesamiento acelerado por GPU**:
  - **Decodificación y redimensionamiento por hardware**: Integración nativa del NVIDIA Video Processing Framework (VPF) a través de `PyNvCodec`. Decodifica, redimensiona y convierte espacios de color directamente en NVDEC.
  - **Detección de escenas**: Implementación personalizada utilizando VPF y OpenCV.
  - **Análisis de audio**: Utiliza `torchaudio` en la GPU para un cálculo rápido del RMS y el flujo espectral.
  - **Análisis de video**: Transmisión de memoria de GPU sin copias (zero-copy) para una estimación de movimiento estable (reemplaza los pesados índices de fotogramas).
  - **Procesamiento de imágenes**: Operadores nativos de PyTorch utilizados para operaciones pesadas como desenfocar fondos (convoluciones separables).
  - **Renderizado**: Motor personalizado de PyTorch+NVENC para un renderizado de alto rendimiento (se ha eliminado MoviePy de la ruta de renderizado).
  - **Procesamiento por lotes robusto**: El procesamiento de video se ejecuta en subprocesos totalmente aislados, limpiando por completo los contextos CUDA entre archivos para evitar la fragmentación de la VRAM y fallos por falta de memoria (OOM), especialmente en Docker/WSL.
  - **Manejo preciso de VFR**: Extrae las Marcas de Tiempo de Presentación (PTS) reales directamente de los paquetes de video para evitar la desincronización de audio/video, manejando el metraje con Tasa de Fotogramas Variable (VFR) sin problemas.
- Puntuación de acción de audio + video:
  - Clasificación combinada con pesos ajustables (valores predeterminados: audio 0.6, video 0.4).
- Escenas clasificadas por puntuación de acción combinada en lugar de por duración.
- **Corte inteligente de escenas**:
  - Selecciona preferentemente escenas completas si encajan dentro del límite de tiempo.
  - **Relleno de escenas (Padding)**: Añade un margen de 1,5 segundos al final de las escenas para capturar animaciones de salida y transiciones.
  - **Recorte inteligente**: Para escenas largas, busca momentos "tranquilos" (bajo audio/movimiento) para cortar, evitando finales abruptos.
- Recorte inteligente con fondo desenfocado opcional para metraje no vertical.
- Lógica de reintentos durante el renderizado para evitar fallos espurios.
- Configuración mediante variables de entorno en un archivo `.env`.

## 📋 Requisitos

- **GPU NVIDIA** con soporte para CUDA.
- **Controladores NVIDIA** (se recomiendan compatibles con CUDA 13.0+).
- Python 3.12+
- FFmpeg (usado para la extracción de audio y codificación NVENC).
- Bibliotecas del sistema: `libgl1`, `libglib2.0-0` (a menudo necesarias para las bibliotecas de visión).

Dependencias de Python (ver `pyproject.toml`):
- `torch`, `torchaudio` (con soporte para CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Instalación

### A través de PyPI (Recomendado)

Asegúrese de tener instalados los controladores NVIDIA y el toolkit de CUDA. Luego instale el paquete directamente:

```bash
pip install shorts-maker-gpu
```

### Configuración manual desde el código fuente (Linux con CUDA)

Asegúrese de tener instalados los controladores NVIDIA y el toolkit de CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Instala la biblioteca y sus dependencias
pip install -e .
```

Si tiene problemas porque PyTorch no detecta la GPU, consulte su guía de instalación para su versión específica de CUDA.

## 💡 Uso

1. Coloque los videos de origen dentro del directorio `gameplay/`.
2. Ejecute la herramienta CLI:

```bash
shorts-maker process
```

Opcionalmente, puede personalizar los directorios de entrada y salida, así como los límites de las escenas:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Los clips generados se guardan en el directorio `generated/`.

Durante el procesamiento, el registro de eventos (log) muestra una puntuación de acción para cada escena combinada y la lista final ordenada por dicha puntuación. Las mejores escenas (por intensidad de acción) se renderizan primero utilizando NVENC.

## 🐳 Docker (Recomendado)

La forma más sencilla de ejecutar esta aplicación es utilizando Docker con el NVIDIA Container Toolkit.

**Requisito previo**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) debe estar instalado en el sistema anfitrión (host).

Construir y ejecutar:

*(Nota: Si la construcción (build) falla con un error de "Segmentation fault" o falta de memoria, limite los núcleos de la CPU usando `docker build --cpuset-cpus="0,1" -t shorts-maker .` en su lugar).*

```bash
docker build -t shorts-maker .

# Ejecutar con acceso a la GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

Observe el flag `--gpus all`, el cual es esencial para que la aplicación acceda a la aceleración por hardware.

## ⚙️ Configuración

Copie `.env.example` a `.env` y ajuste los valores según sea necesario.

Variables compatibles (se muestran los valores predeterminados):
- `TARGET_RATIO_W=9` — Parte del ancho de la relación de aspecto objetivo (ej. 9 para 9:16).
- `TARGET_RATIO_H=16` — Parte del alto de la relación de aspecto objetivo (ej. 16 para 9:16).
- `SCENE_LIMIT=4` — Número máximo de las mejores escenas a renderizar por video de origen.
- `SCENE_THRESHOLD=45.0` — Umbral para los cortes de detección de escenas.
- `X_CENTER=0.5` — Centro del recorte horizontal en el rango [0.0, 1.0].
- `Y_CENTER=0.5` — Centro del recorte vertical en el rango [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Profundidad máxima de reintentos si el renderizado falla.
- `MIN_SHORT_LENGTH=15` — Longitud mínima del short en segundos.
- `MAX_SHORT_LENGTH=179` — Longitud máxima del short en segundos.
- `MAX_COMBINED_SCENE_LENGTH=300` — Longitud combinada máxima (en segundos).
- `SKIP_FIRST_SECONDS=0.0` — Segundos a omitir desde el inicio del video (útil para saltar las pantallas de introducción).
- `SAVE_FFMPEG_LOGS=False` — Determina si se deben guardar los registros de FFmpeg durante el renderizado.
- `LOG_LEVEL=WARNING` — Nivel de registro de eventos (ej. INFO, DEBUG, WARNING).

## 🛠️ Desarrollo

### Linting

Este proyecto usa `ruff` para un linting rápido.

```bash
pip install ruff
ruff check .
```

## 🧪 Ejecución de pruebas

Las pruebas unitarias se encuentran en la carpeta `tests/`. Ejecútelas con:

```bash
pytest -q
```

Nota: Las pruebas están diseñadas para simular (mock) la disponibilidad de la GPU en caso de que falte, de modo que puedan ejecutarse en entornos de integración continua (CI) estándar.

## 🚑 Solución de problemas

- **"internal compiler error: Segmentation fault" durante `docker build`**: Esto ocurre típicamente debido a un error de falta de memoria (OOM) cuando Docker intenta compilar bibliotecas C++/CUDA pesadas (como VPF) usando todos los núcleos de CPU disponibles. Para solucionarlo, limite el número de núcleos de CPU utilizados durante el proceso de construcción:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternativamente, puede aumentar el límite de RAM para Docker/WSL2 en la configuración de su sistema).*
- **"WSL integration with distro unexpectedly stopped" / OOM durante `docker run`**: Procesar video en alta resolución puede consumir una cantidad significativa de RAM/VRAM, lo que provoca que la máquina virtual de WSL2 se bloquee por un error de falta de memoria (OOM). Para solucionar esto, limite la cantidad de núcleos de CPU que el contenedor puede usar durante la ejecución añadiendo el flag `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: Asegúrese de estar ejecutando el contenedor Docker con `--gpus all` o de tener instalado localmente el toolkit de CUDA correcto.
- **Error de NVENC**: Si `h264_nvenc` falla, el script intentará recurrir a la codificación por software (`libx264`). Verifique si su GPU soporta NVENC y si los controladores están actualizados.

## 📄 Licencia

Este proyecto se publica bajo la [Licencia MIT](LICENSE).