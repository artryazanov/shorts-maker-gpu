> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (Optimizado para GPU)

Shorts Maker genera clips de video verticales a partir de videos de juego (gameplay) más largos. Esta biblioteca de Python y herramienta de línea de comandos (CLI) detecta escenas, calcula perfiles de acción de audio y video (intensidad del sonido + movimiento visual) y los combina para clasificar las escenas por su intensidad general. Luego, recorta a la relación de aspecto deseada y renderiza *shorts* listos para subir.

**Esta versión ha sido fuertemente optimizada para GPUs NVIDIA utilizando CUDA.**

Para la versión original solo para CPU, por favor visita [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

### [Leer la Documentación Completa 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ Características

- **Procesamiento Acelerado por GPU**:
  - **Decodificación y Redimensionamiento por Hardware**: Integración nativa del NVIDIA Video Processing Framework (VPF) a través de `PyNvCodec`. Decodifica, redimensiona y convierte espacios de color directamente en NVDEC.
  - **Detección de Escenas**: Implementación personalizada utilizando VPF y OpenCV.
  - **Análisis de Audio**: Utiliza `torchaudio` en la GPU para el cálculo rápido del valor RMS y el flujo espectral.
  - **Análisis de Video**: Transmisión de memoria de GPU sin copias (zero-copy) para una estimación de movimiento estable (reemplaza los pesados índices de fotogramas).
  - **Procesamiento de Imágenes**: Operadores nativos de PyTorch utilizados para operaciones pesadas como desenfocar fondos (convoluciones separables).
  - **Renderizado**: Motor personalizado de PyTorch+NVENC para un renderizado de alto rendimiento (se eliminó MoviePy de la ruta de renderizado).
  - **Procesamiento por Lotes Robusto**: El procesamiento de video se ejecuta en subprocesos totalmente aislados, limpiando completamente los contextos de CUDA entre archivos para evitar la fragmentación de la VRAM y caídas por falta de memoria (OOM), especialmente en Docker/WSL.
- Puntuación de acción de audio + video:
  - Clasificación combinada con pesos ajustables (valores predeterminados: audio 0.6, video 0.4).
- Escenas clasificadas por puntuación de acción combinada en lugar de duración.
- **Corte de Escenas Inteligente**:
  - Selecciona preferentemente escenas completas si encajan dentro del límite de tiempo.
  - **Relleno de Escenas (Padding)**: Añade un búfer de 1.5 segundos al final de las escenas para capturar animaciones de salida y desvanecimientos.
  - **Recorte Inteligente**: Para escenas largas, busca momentos "tranquilos" (bajo audio/movimiento) para cortar, evitando finales abruptos.
- Recorte inteligente con fondo desenfocado opcional para videos que no son verticales.
- Lógica de reintento durante el renderizado para evitar fallos espurios.
- Configuración mediante variables de entorno `.env`.

## 📋 Requisitos

- **GPU NVIDIA** con soporte para CUDA.
- **Controladores NVIDIA** (se recomienda compatibilidad con CUDA 13.0+).
- Python 3.12+
- FFmpeg (usado para la extracción de audio y codificación NVENC).
- Bibliotecas del sistema: `libgl1`, `libglib2.0-0` (a menudo necesarias para las bibliotecas de visión).

Dependencias de Python (ver `pyproject.toml`):
- `torch`, `torchaudio` (con soporte para CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Instalación

### Vía PyPI (Recomendado)

Asegúrate de tener instalados los controladores de NVIDIA y el kit de herramientas de CUDA (CUDA toolkit). Luego, instala el paquete directamente:

```bash
pip install shorts-maker-gpu
```

### Configuración Manual desde el Código Fuente (Linux con CUDA)

Asegúrate de tener instalados los controladores de NVIDIA y el kit de herramientas de CUDA.

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

1. Coloca los videos de origen en el directorio `gameplay/`.
2. Ejecuta la herramienta CLI:

```bash
shorts-maker process
```

Opcionalmente, puedes personalizar los directorios de entrada y salida y los límites de las escenas:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Los clips generados se escriben en el directorio `generated/`.

Durante el procesamiento, el registro de eventos muestra una puntuación de acción para cada escena combinada y la lista final ordenada por esa puntuación. Las mejores escenas (por intensidad de acción) se renderizan primero usando NVENC.

## 🐳 Docker (Recomendado)

La forma más sencilla de ejecutar esta aplicación es usando Docker con el NVIDIA Container Toolkit.

**Requisito previo**: El [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) debe estar instalado en el host.

Construir y ejecutar:

*(Nota: Si la compilación falla con un "Segmentation fault" o error de memoria, limita los núcleos de la CPU utilizando `docker build --cpuset-cpus="0,1" -t shorts-maker .` en su lugar).*

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

Ten en cuenta el indicador `--gpus all`, el cual es esencial para que la aplicación acceda a la aceleración por hardware.

## ⚙️ Configuración

Copia `.env.example` a `.env` y ajusta los valores según sea necesario.

Variables compatibles (se muestran los valores predeterminados):
- `TARGET_RATIO_W=9` — Parte de la anchura de la relación de aspecto deseada (ej., 9 para 9:16).
- `TARGET_RATIO_H=16` — Parte de la altura de la relación de aspecto deseada (ej., 16 para 9:16).
- `SCENE_LIMIT=4` — Número máximo de mejores escenas renderizadas por video de origen.
- `SCENE_THRESHOLD=45.0` — Umbral para cortes de detección de escenas.
- `X_CENTER=0.5` — Centro del recorte horizontal en el rango [0.0, 1.0].
- `Y_CENTER=0.5` — Centro del recorte vertical en el rango [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Profundidad máxima de reintentos si el renderizado falla.
- `MIN_SHORT_LENGTH=15` — Longitud mínima del short en segundos.
- `MAX_SHORT_LENGTH=179` — Longitud máxima del short en segundos.
- `MAX_COMBINED_SCENE_LENGTH=300` — Longitud máxima combinada (en segundos).
- `SAVE_FFMPEG_LOGS=False` — Define si se deben guardar los registros de FFmpeg durante el renderizado.
- `LOG_LEVEL=WARNING` — Nivel de registro (ej., INFO, DEBUG, WARNING).

## 🛠️ Desarrollo

### Linting

Este proyecto utiliza `ruff` para un linting rápido.

```bash
pip install ruff
ruff check .
```

## 🧪 Ejecución de Pruebas

Las pruebas unitarias se encuentran en la carpeta `tests/`. Ejecútalas con:

```bash
pytest -q
```

Nota: Las pruebas están diseñadas para simular la disponibilidad de la GPU si esta no se encuentra, por lo que pueden ejecutarse en entornos de integración continua (CI) estándar.

## 🚑 Solución de Problemas

- **"internal compiler error: Segmentation fault" durante `docker build`**: Esto suele ocurrir debido a un error de falta de memoria (OOM - Out-Of-Memory) cuando Docker intenta compilar pesadas bibliotecas de C++/CUDA (como VPF) usando todos los núcleos de CPU disponibles. Para solucionarlo, limita el número de núcleos de CPU usados durante la compilación:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternativamente, puedes aumentar el límite de RAM para Docker/WSL2 en la configuración de tu sistema).*
- **"Torch not installed" / "CUDA not available"**: Asegúrate de estar ejecutando dentro del contenedor Docker con `--gpus all` o de tener instalado localmente el kit de herramientas CUDA correcto.
- **Error de NVENC**: Si `h264_nvenc` falla, el script intentará recurrir a la codificación por software (`libx264`). Comprueba si tu GPU soporta NVENC y si los controladores están actualizados.

## 📄 Licencia

Este proyecto se distribuye bajo la [Licencia MIT](LICENSE).