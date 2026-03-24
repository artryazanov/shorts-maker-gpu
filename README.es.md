> 🌐 **Idiomas:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (Optimizado para GPU)

Shorts Maker genera videoclips verticales a partir de grabaciones de juego (gameplays) más largas. Esta biblioteca de Python y herramienta CLI detecta escenas, calcula los perfiles de acción de audio y video (intensidad del sonido + movimiento visual) y los combina para clasificar las escenas según su intensidad general. Luego recorta la imagen a la relación de aspecto deseada y renderiza "shorts" listos para subir.

**Esta versión ha sido altamente optimizada para GPUs de NVIDIA utilizando CUDA.**

Para la versión original exclusiva para CPU, por favor visita [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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
  - **Decodificación y redimensionado por hardware**: Integración nativa del NVIDIA Video Processing Framework (VPF) mediante `PyNvCodec`. Decodifica, redimensiona y convierte espacios de color directamente en NVDEC.
  - **Detección de escenas**: Implementación personalizada utilizando VPF y OpenCV.
  - **Análisis de audio**: Utiliza `torchaudio` en la GPU para un cálculo rápido del valor RMS y el flujo espectral.
  - **Análisis de video**: Transmisión en memoria de GPU con cero copias (zero-copy) para una estimación de movimiento estable (reemplaza los pesados índices de fotogramas).
  - **Procesamiento de imágenes**: Uso de operadores nativos de PyTorch para operaciones pesadas como desenfoque de fondos (convoluciones separables).
  - **Renderizado**: Motor personalizado de PyTorch+NVENC para un renderizado de alto rendimiento (se ha eliminado MoviePy de la ruta de renderizado).
  - **Procesamiento por lotes robusto**: El procesamiento de video se ejecuta en subprocesos completamente aislados, limpiando por completo los contextos de CUDA entre archivos para evitar la fragmentación de la VRAM y caídas por falta de memoria (OOM), especialmente en Docker/WSL.
- Puntuación de acción de audio + video:
  - Clasificación combinada con pesos ajustables (valores por defecto: audio 0.6, video 0.4).
- Escenas clasificadas según la puntuación de acción combinada en lugar de su duración.
- **Corte inteligente de escenas**:
  - Selecciona preferentemente escenas completas si se ajustan al límite de tiempo.
  - **Relleno de escena (Padding)**: Añade un margen de 1.5 segundos al final de las escenas para capturar animaciones de salida y desvanecimientos (fades).
  - **Recorte inteligente (Trimming)**: En escenas largas, busca momentos "tranquilos" (bajo audio/movimiento) para realizar cortes, evitando finales abruptos.
- Recorte inteligente con fondo desenfocado opcional para material que no es vertical.
- Lógica de reintento durante el renderizado para evitar fallos espurios.
- Configuración mediante variables de entorno en el archivo `.env`.

## 📋 Requisitos

- **GPU NVIDIA** compatible con CUDA.
- **Controladores de NVIDIA** (se recomiendan compatibles con CUDA 13.0+).
- Python 3.12+
- FFmpeg (usado para la extracción de audio y codificación NVENC).
- Bibliotecas del sistema: `libgl1`, `libglib2.0-0` (a menudo necesarias para bibliotecas de visión por computadora).

Dependencias de Python (ver `pyproject.toml`):
- `torch`, `torchaudio` (compatibles con CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Instalación

### A través de PyPI (Recomendado)

Asegúrate de tener instalados los controladores de NVIDIA y el toolkit de CUDA. Luego instala el paquete directamente:

```bash
pip install shorts-maker-gpu
```

### Configuración manual desde el código fuente (Linux con CUDA)

Asegúrate de tener instalados los controladores de NVIDIA y el toolkit de CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Instala la biblioteca y sus dependencias
pip install -e .
```

Si encuentras problemas con PyTorch y no detecta la GPU, consulta su guía de instalación para tu versión específica de CUDA.

## 💡 Uso

1. Coloca los videos de origen dentro del directorio `gameplay/`.
2. Ejecuta la herramienta CLI:

```bash
shorts-maker process
```

Opcionalmente, puedes personalizar los directorios de entrada y salida, así como los límites de escenas:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Los clips generados se guardan en el directorio `generated/`.

Durante el procesamiento, el registro (log) muestra una puntuación de acción para cada escena combinada y la lista final ordenada por dicha puntuación. Las mejores escenas (por intensidad de acción) se renderizan primero utilizando NVENC.

## 🐳 Docker (Recomendado)

La forma más sencilla de ejecutar esta aplicación es usando Docker con el NVIDIA Container Toolkit.

**Requisito previo**: El [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) debe estar instalado en el sistema anfitrión.

Construye y ejecuta:

*(Nota: Si la construcción (build) falla con un "Segmentation fault" (Fallo de segmentación) o error de memoria, limita los núcleos de CPU utilizando `docker build --cpuset-cpus="0,1" -t shorts-maker .` en su lugar).*

```bash
docker build -t shorts-maker .

# Ejecuta con acceso a la GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

Observa la opción `--gpus all`, que es esencial para que la aplicación acceda a la aceleración por hardware.

## ⚙️ Configuración

Copia `.env.example` a `.env` y ajusta los valores según lo necesites.

Variables soportadas (se muestran los valores por defecto):
- `TARGET_RATIO_W=9` — Parte del ancho en la relación de aspecto objetivo (ej., 9 para 9:16).
- `TARGET_RATIO_H=16` — Parte de la altura en la relación de aspecto objetivo (ej., 16 para 9:16).
- `SCENE_LIMIT=4` — Número máximo de mejores escenas a renderizar por video de origen.
- `SCENE_THRESHOLD=45.0` — Umbral para los cortes de detección de escenas.
- `X_CENTER=0.5` — Centro de recorte horizontal en el rango [0.0, 1.0].
- `Y_CENTER=0.5` — Centro de recorte vertical en el rango [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Profundidad máxima de reintentos si el renderizado falla.
- `MIN_SHORT_LENGTH=15` — Longitud mínima del short en segundos.
- `MAX_SHORT_LENGTH=179` — Longitud máxima del short en segundos.
- `MAX_COMBINED_SCENE_LENGTH=300` — Longitud combinada máxima (en segundos).
- `SAVE_FFMPEG_LOGS=False` — Indica si se guardan los registros de FFmpeg durante el renderizado.
- `LOG_LEVEL=WARNING` — Nivel de registro (ej., INFO, DEBUG, WARNING).

## 🛠️ Desarrollo

### Linting

Este proyecto usa `ruff` para un linting rápido.

```bash
pip install ruff
ruff check .
```

## 🧪 Ejecución de Pruebas

Las pruebas unitarias se encuentran en la carpeta `tests/`. Ejecútalas con:

```bash
pytest -q
```

Nota: Las pruebas están diseñadas para simular (mock) la disponibilidad de la GPU si esta no se encuentra, por lo que pueden ejecutarse en entornos de integración continua (CI) estándar.

## 🚑 Solución de Problemas

- **"internal compiler error: Segmentation fault" durante `docker build`**: Esto ocurre normalmente debido a un error de falta de memoria (OOM) cuando Docker intenta compilar bibliotecas pesadas de C++/CUDA (como VPF) usando todos los núcleos de CPU disponibles. Para solucionarlo, limita el número de núcleos de CPU usados durante el proceso de construcción:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternativamente, puedes aumentar el límite de memoria RAM para Docker/WSL2 en la configuración de tu sistema).*
- **"Torch not installed" / "CUDA not available"**: Asegúrate de estar ejecutando dentro del contenedor de Docker con `--gpus all` o de tener el toolkit de CUDA correcto instalado localmente.
- **Error de NVENC**: Si `h264_nvenc` falla, el script intentará usar como respaldo la codificación por software (`libx264`). Comprueba si tu GPU soporta NVENC y si los controladores están actualizados.

## 📄 Licencia

Este proyecto se publica bajo la [Licencia MIT](LICENSE).