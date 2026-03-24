> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (Оптимизировано для GPU)

Shorts Maker генерирует вертикальные видеоролики из длинных записей геймплея. Эта Python-библиотека и CLI-инструмент обнаруживает сцены, вычисляет профили аудио и видео активности (интенсивность звука + визуальное движение) и объединяет их для ранжирования сцен по общей интенсивности. Затем он обрезает видео под нужное соотношение сторон и рендерит готовые к загрузке shorts.

**Эта версия сильно оптимизирована для графических процессоров NVIDIA с использованием CUDA.**

Для оригинальной версии только для CPU, пожалуйста, посетите [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

- **Ускорение обработки на GPU**:
  - **Аппаратное декодирование и изменение размера**: Нативная интеграция NVIDIA Video Processing Framework (VPF) через `PyNvCodec`. Декодирует, изменяет размер и конвертирует цветовые пространства непосредственно на NVDEC.
  - **Обнаружение сцен**: Кастомная реализация с использованием VPF и OpenCV.
  - **Анализ аудио**: Использует `torchaudio` на GPU для быстрого расчета RMS и спектрального потока.
  - **Анализ видео**: Потоковая передача в памяти GPU без копирования (zero-copy) для стабильной оценки движения (заменяет тяжелые индексы кадров).
  - **Обработка изображений**: Нативные операторы PyTorch используются для тяжелых операций, таких как размытие фона (сепарабельные свертки).
  - **Рендеринг**: Кастомный движок PyTorch+NVENC для высокопроизводительного рендеринга (MoviePy удален из процесса рендеринга).
  - **Надежная пакетная обработка**: Обработка видео выполняется в полностью изолированных подпроцессах, полностью очищая контексты CUDA между файлами, чтобы предотвратить фрагментацию VRAM и сбои OOM (особенно в Docker/WSL).
- Оценка аудио + видео активности:
  - Комбинированное ранжирование с настраиваемыми весами (по умолчанию: аудио 0.6, видео 0.4).
- Сцены ранжируются по комбинированной оценке активности, а не по длительности.
- **Умная нарезка сцен**:
  - Предпочтительно выбирает полные сцены, если они вписываются в лимит времени.
  - **Отступы сцен**: Добавляет 1,5-секундный буфер в конец сцен для захвата анимаций выхода и затуханий.
  - **Умная обрезка**: Для длинных сцен ищет «тихие» моменты (низкий уровень аудио/движения) для обрезки, избегая резких обрывов.
- Умное кадрирование с опциональным размытым фоном для невертикальных видео.
- Логика повторных попыток во время рендеринга во избежание случайных сбоев.
- Настройка через переменные окружения `.env`.

## 📋 Требования

- **NVIDIA GPU** с поддержкой CUDA.
- **Драйверы NVIDIA** (рекомендуется совместимость с CUDA 13.0+).
- Python 3.12+
- FFmpeg (используется для извлечения аудио и кодирования NVENC).
- Системные библиотеки: `libgl1`, `libglib2.0-0` (часто требуются для библиотек компьютерного зрения).

Зависимости Python (см. `pyproject.toml`):
- `torch`, `torchaudio` (с поддержкой CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 Установка

### Через PyPI (Рекомендуется)

Убедитесь, что у вас установлены драйверы NVIDIA и набор инструментов CUDA. Затем установите пакет напрямую:

```bash
pip install shorts-maker-gpu
```

### Ручная установка из исходного кода (Linux с CUDA)

Убедитесь, что у вас установлены драйверы NVIDIA и набор инструментов CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Установка библиотеки и её зависимостей
pip install -e .
```

Если вы столкнулись с проблемой, когда PyTorch не находит GPU, обратитесь к руководству по его установке для вашей конкретной версии CUDA.

## 💡 Использование

1. Поместите исходные видео в директорию `gameplay/`.
2. Запустите CLI-инструмент:

```bash
shorts-maker process
```

Вы можете дополнительно настроить директории ввода и вывода, а также лимиты сцен:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Сгенерированные клипы записываются в директорию `generated/`.

В процессе обработки в логах отображается оценка активности для каждой объединенной сцены и финальный список, отсортированный по этой оценке. Топовые сцены (по интенсивности действий) рендерятся первыми с использованием NVENC.

## 🐳 Docker (Рекомендуется)

Самый простой способ запустить это приложение — использовать Docker вместе с NVIDIA Container Toolkit.

**Предварительное условие**: на хост-машине должен быть установлен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Сборка и запуск:

*(Примечание: Если сборка завершается с ошибкой «Segmentation fault» или ошибкой памяти, ограничьте количество ядер CPU, используя вместо этого `docker build --cpuset-cpus="0,1" -t shorts-maker .`).*

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

Обратите внимание на флаг `--gpus all`, который необходим приложению для доступа к аппаратному ускорению.

## ⚙️ Конфигурация

Скопируйте `.env.example` в `.env` и настройте значения по необходимости.

Поддерживаемые переменные (показаны значения по умолчанию):
- `TARGET_RATIO_W=9` — Значение ширины целевого соотношения сторон (например, 9 для 9:16).
- `TARGET_RATIO_H=16` — Значение высоты целевого соотношения сторон (например, 16 для 9:16).
- `SCENE_LIMIT=4` — Максимальное количество топовых сцен, рендерящихся для каждого исходного видео.
- `X_CENTER=0.5` — Горизонтальный центр обрезки в диапазоне [0.0, 1.0].
- `Y_CENTER=0.5` — Вертикальный центр обрезки в диапазоне [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Максимальная глубина повторных попыток при сбое рендеринга.
- `MIN_SHORT_LENGTH=15` — Минимальная длина шортса в секундах.
- `MAX_SHORT_LENGTH=179` — Максимальная длина шортса в секундах.
- `MAX_COMBINED_SCENE_LENGTH=300` — Максимальная суммарная длительность (в секундах).
- `SAVE_FFMPEG_LOGS=False` — Сохранять ли логи FFmpeg во время рендеринга.

## 🛠️ Разработка

### Линтинг

Этот проект использует `ruff` для быстрого линтинга.

```bash
pip install ruff
ruff check .
```

## 🧪 Запуск тестов

Модульные тесты (Unit tests) находятся в папке `tests/`. Запустите их с помощью:

```bash
pytest -q
```

Примечание: Тесты разработаны так, чтобы имитировать наличие GPU в случае его отсутствия, поэтому они могут запускаться в стандартных CI-окружениях.

## 🚑 Решение проблем

- **"internal compiler error: Segmentation fault" во время `docker build`**: Это обычно происходит из-за ошибки нехватки памяти (OOM), когда Docker пытается скомпилировать тяжелые C++/CUDA библиотеки (например, VPF), используя все доступные ядра CPU. Чтобы исправить это, ограничьте количество ядер CPU, используемых в процессе сборки:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(В качестве альтернативы вы можете увеличить лимит оперативной памяти для Docker/WSL2 в настройках вашей системы).*
- **"Torch not installed" / "CUDA not available"**: Убедитесь, что вы работаете внутри Docker-контейнера с флагом `--gpus all` или что у вас локально установлен правильный набор инструментов CUDA.
- **Ошибка NVENC**: Если `h264_nvenc` завершается сбоем, скрипт попытается переключиться на программное кодирование (`libx264`). Проверьте, поддерживает ли ваш GPU NVENC и обновлены ли драйверы.

## 📄 Лицензия

Этот проект выпущен под [лицензией MIT](LICENSE).