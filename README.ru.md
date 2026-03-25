> 🌐 **Языки:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (Оптимизировано для GPU)

Shorts Maker создает короткие вертикальные видео из длинных геймплейных роликов. Эта Python-библиотека и утилита командной строки (CLI) определяет сцены, вычисляет профили аудио- и видеоактивности (интенсивность звука + визуальное движение) и комбинирует их для ранжирования сцен по общей интенсивности. Затем она обрезает видео до нужного соотношения сторон и рендерит готовые к публикации Shorts.

**Эта версия была глубоко оптимизирована для GPU NVIDIA с использованием CUDA.**

Оригинальная версия, работающая только на CPU, доступна по ссылке [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

## ✨ Особенности

- **Обработка с аппаратным ускорением на GPU**:
  - **Аппаратное декодирование и изменение размера**: Нативная интеграция NVIDIA Video Processing Framework (VPF) через `PyNvCodec`. Декодирует, изменяет размер и преобразует цветовые пространства прямо на NVDEC.
  - **Распознавание сцен**: Кастомная реализация с использованием VPF и OpenCV.
  - **Анализ аудио**: Использует `torchaudio` на GPU для быстрого вычисления RMS и спектрального потока.
  - **Анализ видео**: Потоковая передача данных в памяти GPU без копирования (zero-copy) для стабильной оценки движения (заменяет тяжеловесные индексы кадров).
  - **Обработка изображений**: Нативные операторы PyTorch используются для тяжелых операций, таких как размытие фона (разделяемые свертки).
  - **Рендеринг**: Кастомный движок на базе PyTorch+NVENC для высокопроизводительного рендеринга (MoviePy удален из цепочки рендеринга).
  - **Надежная пакетная обработка**: Обработка видео запускается в полностью изолированных подпроцессах, полностью очищая контексты CUDA между файлами для предотвращения фрагментации VRAM и сбоев из-за нехватки памяти (OOM), особенно в Docker/WSL.
- Оценка активности по аудио и видео:
  - Комбинированное ранжирование с настраиваемыми весами (по умолчанию: аудио 0.6, видео 0.4).
- Сцены ранжируются по комбинированной оценке активности, а не по продолжительности.
- **Умная нарезка сцен**:
  - Предпочтительно выбирает целые сцены, если они вписываются в лимит времени.
  - **Отступы сцен (Padding)**: Добавляет 1.5-секундный буфер в конец сцен для захвата завершающих анимаций и затуханий.
  - **Умная обрезка**: Для длинных сцен ищет "тихие" моменты (низкая активность аудио/движения) для обрезки, избегая резких обрывов.
- Умное кадрирование с опциональным размытым фоном для невертикальных исходников.
- Логика повторных попыток при рендеринге для предотвращения случайных сбоев.
- Конфигурация через переменные окружения в файле `.env`.

## 📋 Требования

- **GPU NVIDIA** с поддержкой CUDA.
- **Драйверы NVIDIA** (рекомендуются совместимые с CUDA 13.0+).
- Python 3.12+
- FFmpeg (используется для извлечения аудио и кодирования NVENC).
- Системные библиотеки: `libgl1`, `libglib2.0-0` (часто требуются для библиотек компьютерного зрения).

Зависимости Python (см. `pyproject.toml`):
- `torch`, `torchaudio` (с поддержкой CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Установка

### Через PyPI (Рекомендуется)

Убедитесь, что у вас установлены драйверы NVIDIA и инструментарий CUDA (CUDA toolkit). Затем установите пакет напрямую:

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

# Установка библиотеки и ее зависимостей
pip install -e .
```

Если вы столкнулись с проблемами, когда PyTorch не находит GPU, обратитесь к руководству по его установке для вашей конкретной версии CUDA.

## 💡 Использование

1. Поместите исходные видео в директорию `gameplay/`.
2. Запустите CLI-утилиту:

```bash
shorts-maker process
```

При желании вы можете настроить директории ввода и вывода, а также лимиты сцен:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Сгенерированные клипы сохраняются в директорию `generated/`.

Во время обработки в логах отображается оценка активности (action score) для каждой комбинированной сцены, а также итоговый список, отсортированный по этой оценке. Топовые сцены (по интенсивности действий) рендерятся в первую очередь с использованием NVENC.

## 🐳 Docker (Рекомендуется)

Самый простой способ запустить это приложение — использовать Docker с NVIDIA Container Toolkit.

**Предварительное требование**: На хост-машине должен быть установлен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Сборка и запуск:

*(Примечание: Если сборка прерывается с ошибкой "Segmentation fault" или ошибкой памяти, ограничьте использование ядер CPU, запустив `docker build --cpuset-cpus="0,1" -t shorts-maker .` вместо обычной команды).*

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

Скопируйте `.env.example` в `.env` и настройте значения при необходимости.

Поддерживаемые переменные (указаны значения по умолчанию):
- `TARGET_RATIO_W=9` — Ширина целевого соотношения сторон (например, 9 для 9:16).
- `TARGET_RATIO_H=16` — Высота целевого соотношения сторон (например, 16 для 9:16).
- `SCENE_LIMIT=4` — Максимальное количество лучших сцен, рендеринг которых выполняется для каждого исходного видео.
- `SCENE_THRESHOLD=45.0` — Порог для обнаружения смены сцен.
- `X_CENTER=0.5` — Центр обрезки по горизонтали в диапазоне [0.0, 1.0].
- `Y_CENTER=0.5` — Центр обрезки по вертикали в диапазоне [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Максимальное количество повторных попыток при сбое рендеринга.
- `MIN_SHORT_LENGTH=15` — Минимальная длина шортса в секундах.
- `MAX_SHORT_LENGTH=179` — Максимальная длина шортса в секундах.
- `MAX_COMBINED_SCENE_LENGTH=300` — Максимальная комбинированная длина сцены (в секундах).
- `SAVE_FFMPEG_LOGS=False` — Сохранять ли логи FFmpeg во время рендеринга.
- `LOG_LEVEL=WARNING` — Уровень логирования (например, INFO, DEBUG, WARNING).

## 🛠️ Разработка

### Линтинг

Этот проект использует `ruff` для быстрого линтинга.

```bash
pip install ruff
ruff check .
```

## 🧪 Запуск тестов

Модульные тесты находятся в папке `tests/`. Запустите их командой:

```bash
pytest -q
```

Примечание: Тесты разработаны с возможностью имитации (mock) наличия GPU, если он отсутствует, поэтому они могут выполняться в стандартных средах CI.

## 🚑 Устранение неполадок

- **"internal compiler error: Segmentation fault" во время `docker build`**: Обычно это происходит из-за ошибки нехватки памяти (OOM), когда Docker пытается скомпилировать тяжелые библиотеки C++/CUDA (например, VPF), используя все доступные ядра CPU. Чтобы исправить это, ограничьте количество ядер CPU, используемых в процессе сборки:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(В качестве альтернативы вы можете увеличить лимит оперативной памяти для Docker/WSL2 в системных настройках).*
- **"Torch not installed" / "CUDA not available"**: Убедитесь, что вы запускаете приложение внутри Docker-контейнера с параметром `--gpus all` или что у вас локально установлен правильный инструментарий CUDA.
- **Ошибка NVENC**: Если `h264_nvenc` завершается с ошибкой, скрипт попытается переключиться на программное кодирование (`libx264`). Проверьте, поддерживает ли ваш GPU технологию NVENC и актуальны ли ваши драйверы.

## 📄 Лицензия

Этот проект выпущен под [лицензией MIT](LICENSE).