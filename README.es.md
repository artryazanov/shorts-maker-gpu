> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (Optimizado para GPU)

Shorts Maker genera videoclips verticales a partir de grabaciones de juego más largas. Esta biblioteca de Python y herramienta de línea de comandos (CLI) detecta escenas, calcula los perfiles de acción de audio y video (intensidad del sonido + movimiento visual), y los combina para clasificar las escenas según su intensidad general. Luego, recorta el video a la relación de aspecto deseada y renderiza "shorts" listos para ser subidos.

**Esta versión ha sido fuertemente optimizada para GPUs de NVIDIA utilizando CUDA.**

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

- **Procesamiento Acelerado por GPU**:
  - **Decodificación y Redimensionamiento por Hardware**: Integración nativa de NVIDIA Video Processing Framework (VPF) a través de `PyNvCodec`. Decodifica, redimensiona y convierte espacios de color directamente en NVDEC.
  - **Detección de Escenas**: Implementación personalizada utilizando VPF y OpenCV.
  - **Análisis de Audio**: Utiliza `torchaudio` en la GPU para el cálculo rápido de RMS y flujo espectral.
  - **Análisis de Video**: Transmisión de memoria en la GPU con copia cero (zero-copy) para una estimación de movimiento estable (reemplaza los pesados índices de fotogramas).
  - **Procesamiento de Imágenes**: Operadores nativos de PyTorch utilizados para operaciones pesadas como el desenfoque de fondos (convoluciones separables).
  - **Renderizado**: Motor personalizado de PyTorch+NVENC para un renderizado de alto rendimiento (se eliminó MoviePy de la ruta de renderizado).
  - **Procesamiento por Lotes Robusto**: El procesamiento de video se ejecuta en subprocesos totalmente aislados, limpiando por completo los contextos de CUDA entre archivos para evitar la fragmentación de la VRAM y fallos por falta de memoria (OOM), especialmente en Docker/WSL.
- Puntuación de acción de audio + video:
  - Clasificación combinada con pesos ajustables (valores por defecto: audio 0.6, video 0.4).
- Escenas clasificadas por la puntuación de acción combinada en lugar de por su duración.
- **Corte Inteligente de Escenas**:
  - Selecciona preferentemente escenas completas si encajan dentro del límite de tiempo.
  - **Relleno de Escena (Padding)**: Añade un búfer de 1.5 segundos al final de las escenas para capturar animaciones de salida y desvanecimientos.
  - **Recorte Inteligente**: Para escenas largas, busca momentos "tranquilos" (bajo nivel de audio/movimiento) para realizar el corte, evitando finales abruptos.
- Recorte inteligente con fondo desenfocado opcional para material que no es vertical.
- Lógica de reintento durante el renderizado para evitar fallos espurios.
- Configuración a través de variables de entorno `.env`.

## 📋 Requisitos

- **GPU de NVIDIA** con soporte para CUDA.
- **Controladores NVIDIA** (se recomienda compatibilidad con CUDA 13.0+).
- Python 3.12+
- FFmpeg (utilizado para la extracción de audio y codificación NVENC).
- Bibliotecas del sistema: `libgl1`, `libglib2.0-0` (a menudo necesarias para bibliotecas de visión).

Dependencias de Python (ver `pyproject.toml`):
- `torch`, `torchaudio` (con soporte para CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Instalación

### Vía PyPI (Recomendado)

Asegúrese de tener instalados los controladores de NVIDIA y el toolkit de CUDA. Luego, instale el paquete directamente:

```bash
pip install shorts-maker-gpu
```

### Configuración Manual desde el Código Fuente (Linux con CUDA)

Asegúrese de tener instalados los controladores de NVIDIA y el toolkit de CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Instalar la biblioteca y sus dependencias
pip install -e .
```

Si encuentra problemas porque PyTorch no detecta la GPU, consulte su guía de instalación para su versión específica de CUDA.

## 💡 Uso

1. Coloque los videos de origen en el directorio `gameplay/`.
2. Ejecute la herramienta CLI:

```bash
shorts-maker process
```

Opcionalmente, puede personalizar los directorios de entrada y salida, y los límites de las escenas:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Los clips generados se guardan en el directorio `generated/`.

Durante el procesamiento, el registro muestra una puntuación de acción para cada escena combinada y la lista final ordenada por dicha puntuación. Las mejores escenas (por intensidad de acción) se renderizan primero utilizando NVENC.

## 🐳 Docker (Recomendado)

La forma más sencilla de ejecutar esta aplicación es usando Docker con el NVIDIA Container Toolkit.

**Requisito previo**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) debe estar instalado en el sistema anfitrión (host).

Compilar y ejecutar:

*(Nota: Si la compilación falla con un "Segmentation fault" (fallo de segmentación) o un error de memoria, limite los núcleos de la CPU utilizando `docker build --cpuset-cpus="0,1" -t shorts-maker .` en su lugar).*

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

Tenga en cuenta la bandera `--gpus all`, la cual es esencial para que la aplicación acceda a la aceleración por hardware.

## ⚙️ Configuración

Copie `.env.example` a `.env` y ajuste los valores según sea necesario.

Variables compatibles (se muestran los valores predeterminados):
- `TARGET_RATIO_W=9` — Parte del ancho de la relación de aspecto objetivo (por ejemplo, 9 para 9:16).
- `TARGET_RATIO_H=16` — Parte del alto de la relación de aspecto objetivo (por ejemplo, 16 para 9:16).
- `SCENE_LIMIT=4` — Número máximo de las mejores escenas a renderizar por cada video de origen.
- `X_CENTER=0.5` — Centro del recorte horizontal en el rango [0.0, 1.0].
- `Y_CENTER=0.5` — Centro del recorte vertical en el rango [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Profundidad máxima de reintentos si el renderizado falla.
- `MIN_SHORT_LENGTH=15` — Longitud mínima del short en segundos.
- `MAX_SHORT_LENGTH=179` — Longitud máxima del short en segundos.
- `MAX_COMBINED_SCENE_LENGTH=300` — Longitud máxima combinada (en segundos).
- `SAVE_FFMPEG_LOGS=False` — Si se deben guardar los registros (logs) de FFmpeg durante el renderizado.
- `LOG_LEVEL=WARNING` — Nivel de registro (por ejemplo, INFO, DEBUG, WARNING).

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

Nota: Las pruebas están diseñadas para simular (mock) la disponibilidad de la GPU si esta falta, de modo que puedan ejecutarse en entornos CI estándar.

## 🚑 Solución de Problemas

- **"internal compiler error: Segmentation fault" durante `docker build`**: Esto suele ocurrir debido a un error por falta de memoria (Out-Of-Memory u OOM) cuando Docker intenta compilar bibliotecas pesadas de C++/CUDA (como VPF) utilizando todos los núcleos disponibles de la CPU. Para solucionarlo, limite el número de núcleos de CPU utilizados durante el proceso de compilación:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(Alternativamente, puede aumentar el límite de RAM para Docker/WSL2 en la configuración de su sistema).*
- **"Torch not installed" / "CUDA not available"**: Asegúrese de estar ejecutando dentro del contenedor de Docker con `--gpus all` o de tener instalado localmente el toolkit de CUDA correcto.
- **Error de NVENC**: Si `h264_nvenc` falla, el script intenta recurrir a la codificación por software (`libx264`). Compruebe si su GPU es compatible con NVENC y si los controladores están actualizados.

## 📄 Licencia

Este proyecto está publicado bajo la [Licencia MIT](LICENSE).