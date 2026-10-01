> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (Оптимизировано для GPU)

Shorts Maker создает вертикальные видеоролики из длинных записей геймплея. Эта Python-библиотека и CLI-инструмент обнаруживает сцены, вычисляет профили аудио- и видеоактивности (интенсивность звука + визуальное движение) и объединяет их для ранжирования сцен по общей интенсивности. Затем он кадрирует видео до нужного соотношения сторон и рендерит готовые к загрузке shorts.

**Эта версия была глубоко оптимизирована для графических процессоров NVIDIA с использованием CUDA.**

Оригинальную версию, работающую только на процессоре (CPU), можно найти здесь: [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

### [Читать полную документацию 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ Возможности

- **Обработка с аппаратным ускорением GPU**:
  - **Аппаратное декодирование и изменение размера**: Нативная интеграция NVIDIA Video Processing Framework (VPF) через `PyNvCodec`. Декодирует, изменяет размер и конвертирует цветовые пространства непосредственно на NVDEC.
  - **Обнаружение сцен**: Собственная реализация с использованием VPF и OpenCV.
  - **Анализ аудио**: Использует `torchaudio` на GPU для быстрого вычисления RMS и спектрального потока.
  - **Анализ видео**: Потоковая передача данных в памяти GPU без копирования (zero-copy) для стабильной оценки движения (заменяет ресурсоемкие индексы кадров).
  - **Обработка изображений**: Нативные операторы PyTorch используются для тяжелых операций, таких как размытие фона (сепарабельные свертки).
  - **Рендеринг**: Собственный движок PyTorch+NVENC для высокопроизводительного рендеринга (MoviePy удален из процесса рендеринга).
  - **Надежная пакетная обработка**: Обработка видео запускается в полностью изолированных подпроцессах, полностью очищая контексты CUDA между файлами для предотвращения фрагментации VRAM и сбоев OOM (нехватки памяти), особенно в Docker/WSL.
  - **Точная обработка VFR**: Извлекает истинные метки времени (PTS) непосредственно из видеопакетов для предотвращения рассинхронизации аудио/видео, бесшовно обрабатывая геймплей с переменной частотой кадров (VFR).
- Оценка аудио + видео активности:
  - Комбинированное ранжирование с настраиваемыми весами (по умолчанию: аудио 0.6, видео 0.4).
- Сцены ранжируются по комбинированной оценке активности, а не по продолжительности.
- **Умная нарезка сцен**:
  - Предпочтительно выбирает полные сцены, если они укладываются в лимит времени.
  - **Отступы сцен (Padding)**: Добавляет 1.5-секундный буфер в конец сцен для захвата анимаций выхода и затуханий.
  - **Умная обрезка**: Для длинных сцен ищет "тихие" моменты (с низкой активностью звука/движения) для обрезки, избегая резких концовок.
- Умное кадрирование с опциональным размытым фоном для невертикальных видео.
- Логика повторных попыток при рендеринге во избежание случайных сбоев.
- Конфигурация через переменные окружения в файле `.env`.

## 📋 Требования

- **NVIDIA GPU** с поддержкой CUDA.
- **Драйверы NVIDIA** (рекомендуются совместимые с CUDA 13.0+).
- Python 3.12+
- FFmpeg (используется для извлечения аудио и кодирования NVENC).
- Системные библиотеки: `libgl1`, `libglib2.0-0` (часто требуются для библиотек компьютерного зрения).

Зависимости Python (см. `pyproject.toml`):
- `torch`, `torchaudio` (с поддержкой CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Установка

### Через PyPI (рекомендуется)

Убедитесь, что у вас установлены драйверы NVIDIA и инструментарий CUDA. Затем установите пакет напрямую:

```bash
pip install shorts-maker-gpu
```

### Ручная установка из исходников (Linux с CUDA)

Убедитесь, что у вас установлены драйверы NVIDIA и инструментарий CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Установите библиотеку и ее зависимости
pip install -e .
```

Если вы столкнулись с тем, что PyTorch не находит GPU, обратитесь к руководству по установке для вашей конкретной версии CUDA.

## 💡 Использование

1. Поместите исходные видео в директорию `gameplay/`.
2. Запустите инструмент CLI:

```bash
shorts-maker process
```

При желании вы можете настроить директории ввода и вывода, а также лимиты сцен:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Сгенерированные клипы сохраняются в директорию `generated/`.

В процессе обработки в логах отображается оценка активности для каждой скомбинированной сцены и итоговый список, отсортированный по этой оценке. Топовые сцены (по интенсивности действия) рендерятся первыми с использованием NVENC.

## 🐳 Docker (рекомендуется)

Самый простой способ запустить это приложение — использовать Docker с NVIDIA Container Toolkit.

**Предварительное требование**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) должен быть установлен на хосте.

Сборка и запуск:

*(Примечание: Если сборка завершается с ошибкой "Segmentation fault" или нехваткой памяти, ограничьте количество ядер CPU, используя команду `docker build --cpuset-cpus="0,1" -t shorts-maker .`)*.

```bash
docker build -t shorts-maker .

# Запуск с доступом к GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

Обратите внимание на флаг `--gpus all`, он необходим приложению для доступа к аппаратному ускорению.

## ⚙️ Конфигурация

Скопируйте `.env.example` в `.env` и настройте значения по мере необходимости.

Поддерживаемые переменные (указаны значения по умолчанию):
- `TARGET_RATIO_W=9` — Часть ширины в целевом соотношении сторон (например, 9 для 9:16).
- `TARGET_RATIO_H=16` — Часть высоты в целевом соотношении сторон (например, 16 для 9:16).
- `SCENE_LIMIT=4` — Максимальное количество лучших сцен, которые будут отрендерены для каждого исходного видео.
- `SCENE_THRESHOLD=45.0` — Порог для определения разрывов сцен.
- `X_CENTER=0.5` — Центр кадрирования по горизонтали в диапазоне [0.0, 1.0].
- `Y_CENTER=0.5` — Центр кадрирования по вертикали в диапазоне [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Максимальная глубина повторных попыток при сбое рендеринга.
- `MIN_SHORT_LENGTH=15` — Минимальная длина шортса в секундах.
- `MAX_SHORT_LENGTH=179` — Максимальная длина шортса в секундах.
- `MAX_COMBINED_SCENE_LENGTH=300` — Максимальная суммарная длина (в секундах).
- `SKIP_FIRST_SECONDS=0.0` — Количество секунд для пропуска с начала видео (полезно для пропуска заставок).
- `SAVE_FFMPEG_LOGS=False` — Сохранять ли логи FFmpeg во время рендеринга.
- `LOG_LEVEL=WARNING` — Уровень логирования (например, INFO, DEBUG, WARNING).

## 🛠️ Разработка

### Линтинг

В этом проекте используется `ruff` для быстрого линтинга.

```bash
pip install ruff
ruff check .
```

## 🧪 Запуск тестов

Юнит-тесты находятся в папке `tests/`. Запустите их с помощью:

```bash
pytest -q
```

Примечание: Тесты спроектированы так, чтобы имитировать (mock) наличие GPU, если он отсутствует, поэтому их можно запускать в стандартных средах CI.

## 🚑 Устранение неполадок

- **"internal compiler error: Segmentation fault" во время `docker build`**: Обычно это происходит из-за ошибки нехватки памяти (OOM), когда Docker пытается скомпилировать тяжелые C++/CUDA библиотеки (например, VPF), используя все доступные ядра CPU. Чтобы это исправить, ограничьте количество ядер CPU, используемых в процессе сборки:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(В качестве альтернативы можно увеличить лимит оперативной памяти для Docker/WSL2 в настройках системы).*
- **"WSL integration with distro unexpectedly stopped" / OOM во время `docker run`**: Обработка видео высокого разрешения может потреблять значительный объем RAM/VRAM, что приводит к сбою виртуальной машины WSL2 из-за ошибки нехватки памяти (OOM). Чтобы это исправить, ограничьте количество ядер CPU, которые контейнер может использовать во время выполнения, добавив флаг `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: Убедитесь, что вы запускаете приложение внутри Docker-контейнера с флагом `--gpus all` или что у вас локально установлен правильный инструментарий CUDA.
- **Ошибка NVENC**: Если `h264_nvenc` завершается с ошибкой, скрипт попытается вернуться к программному кодированию (`libx264`). Проверьте, поддерживает ли ваш GPU кодировщик NVENC и обновлены ли драйверы.

## 📄 Лицензия

Этот проект выпущен под лицензией [MIT License](LICENSE).