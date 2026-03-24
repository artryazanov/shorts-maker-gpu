> 🌐 **Idiomas:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (Optimizado para GPU)

Shorts Maker genera clips de video verticales a partir de grabaciones de juego más largas. Esta biblioteca de Python y herramienta de línea de comandos (CLI) detecta escenas, calcula perfiles de acción de audio y video (intensidad de sonido + movimiento visual) y los combina para clasificar las escenas según su intensidad general. Luego, recorta al formato de relación de aspecto deseado y renderiza "shorts" listos para subir.

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

### [Leer la Documentación Completa 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ Características

- **Procesamiento Acelerado por GPU**:
  - **Decodificación y Redimensionamiento por Hardware**: Integración nativa con NVIDIA Video Processing Framework (VPF) mediante `PyNvCodec`. Decodifica, redimensiona y convierte espacios de color directamente en NVDEC.
  - **Detección de Escenas**: Implementación personalizada utilizando VPF y OpenCV.
  - **Análisis de Audio**: Utiliza `torchaudio` en la GPU para el cálculo rápido de RMS y flujo espectral.
  - **Análisis de Video**: Transmisión de memoria GPU sin copias (zero-copy) para una estimación de movimiento estable (reemplaza los pesados índices de fotogramas).
  - **Procesamiento de Imágenes**: Operadores nativos de PyTorch utilizados para operaciones pesadas como desenfocar fondos (convoluciones separables).
  - **Renderizado**: Motor personalizado de PyTorch+NVENC para renderizado de alto rendimiento (Se ha eliminado MoviePy de la ruta de renderizado).
  - **Procesamiento por Lotes Robusto**: El procesamiento de video se ejecuta en subprocesos completamente aislados, limpiando por completo los contextos de CUDA entre archivos para prevenir la fragmentación de VRAM y caídas por falta de memoria (OOM), especialmente en Docker/WSL.
- Puntuación de acción de audio + video:
  - Clasificación combinada con pesos ajustables (por defecto: audio 0.6, video 0.4).
- Las escenas se clasifican por su puntuación de acción combinada en lugar de por su duración.
- **Corte Inteligente de Escenas**:
  - Selecciona preferentemente escenas completas si encajan dentro del límite de tiempo.
  - **Relleno de Escenas (Padding)**: Añade un margen de 1.5 segundos al final de las escenas para capturar animaciones de salida y desvanecimientos.
  - **Recorte Inteligente**: En escenas largas, busca momentos "tranquilos" (bajo nivel de audio/movimiento) para realizar el corte, evitando finales abruptos.
- Recorte inteligente con fondo desenfocado opcional para grabaciones no verticales.
- Lógica de reintentos durante el renderizado para evitar fallos espurios.
- Configuración a través de variables de entorno en el archivo `.env`.

## 📋 Requisitos

- **GPU NVIDIA** con soporte para CUDA.
- **Controladores NVIDIA** (se recomienda compatibilidad con CUDA 13.0+).
- Python 3.12+
- FFmpeg (utilizado para la extracción de audio y codificación NVENC).
- Bibliotecas del sistema: `libgl1`, `libglib2.0-0` (a menudo necesarias para bibliotecas de visión).

Dependencias de Python (ver `pyproject.toml`):
- `torch`, `torchaudio` (con soporte para CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Instalación

### Vía PyPI (Recomendado)

Asegúrese de tener instalados los controladores NVIDIA y el toolkit de CUDA. Luego, instale el paquete directamente:

```bash
pip install shorts-maker-gpu
```

### Configuración Manual desde el Código Fuente (Linux con CUDA)

Asegúrese de tener instalados los controladores NVIDIA y el toolkit de CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Instalar la biblioteca y sus dependencias
pip install -e .
```

Si tiene problemas porque PyTorch no encuentra la GPU, consulte su guía de instalación para su versión específica de CUDA.

## 💡 Uso

1. Coloque los videos de origen dentro del directorio `gameplay/`.
2. Ejecute la herramienta CLI:

```bash
shorts-maker process
```

Opcionalmente, puede personalizar los directorios de entrada y salida, así como los límites de escena:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Los clips generados se guardarán en el directorio `generated/`.

Durante el procesamiento, el registro muestra una puntuación de acción para cada escena combinada y la lista final ordenada por dicha puntuación. Las mejores escenas (por intensidad de acción) se renderizan primero usando NVENC.

## 🐳 Docker (Recomendado)

La forma más fácil de ejecutar esta aplicación es usando Docker con el NVIDIA Container Toolkit.

**Requisito previo**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) debe estar instalado en el sistema host.

Construir y ejecutar:

*(Nota: Si la construcción falla con un error de segmentación ("Segmentation fault") o error de memoria, limite los núcleos de CPU usando `docker build --cpuset-cpus="0,1" -t shorts-maker .` en su lugar).*

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

Observe la bandera `--gpus all`, la cual es esencial para que la aplicación acceda a la aceleración por hardware.

## ⚙️ Configuración

Copie `.env.example` a `.env` y ajuste los valores según sea necesario.

Variables compatibles (se muestran los valores por defecto):
- `TARGET_RATIO_W=9` — Parte del ancho de la relación de aspecto objetivo (ej., 9 para 9:16).
- `TARGET_RATIO_H=16` — Parte del alto de la relación de aspecto objetivo (ej., 16 para 9:16).
- `SCENE_LIMIT=4` — Número máximo de mejores escenas renderizadas por video de origen.
- `X_CENTER=0.5` — Centro del recorte horizontal en el rango [0.0, 1.0].
- `Y_CENTER=0.5` — Centro del recorte vertical en el rango [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Profundidad máxima de reintentos si falla el renderizado.
- `MIN_SHORT_LENGTH=15` — Duración mínima del corto en segundos.
- `MAX_SHORT_LENGTH=179` — Duración máxima del corto en segundos.
- `MAX_COMBINED_SCENE_LENGTH=300` — Duración combinada máxima (en segundos).
- `SAVE_FFMPEG_LOGS=False` — Si se deben guardar los registros (logs) de FFmpeg durante el renderizado.

## 🛠️ Desarrollo

### Linting

Este proyecto utiliza `ruff` para un linting rápido.

```bash
pip install ruff
ruff check .
```

## 🧪 Ejecución de Pruebas

Las pruebas unitarias se encuentran en la carpeta `tests/`. Ejecútelas con:

```bash
pytest -q
```

Nota: Las pruebas están diseñadas para simular (mock) la disponibilidad de la GPU si falta, de modo que puedan ejecutarse en entornos CI estándar.

## 🚑 Solución de Problemas

- **"internal compiler error: Segmentation fault" durante `docker build`**: Esto suele ocurrir debido a un error de falta de memoria (OOM) cuando Docker intenta compilar bibliotecas pesadas de C++/CUDA (como VPF) utilizando todos los núcleos de CPU disponibles. Para solucionarlo, limite el número de núcleos de CPU utilizados durante el proceso de compilación:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternativamente, puede aumentar el límite de RAM para Docker/WSL2 en la configuración de su sistema).*
- **"Torch not installed" / "CUDA not available"**: Asegúrese de estar ejecutando dentro del contenedor de Docker con `--gpus all` o de tener instalado localmente el toolkit de CUDA correcto.
- **NVENC Error**: Si `h264_nvenc` falla, el script intentará usar como alternativa la codificación por software (`libx264`). Verifique si su GPU soporta NVENC y si los controladores están actualizados.

## 📄 Licencia

Este proyecto se publica bajo la [Licencia MIT](LICENSE).