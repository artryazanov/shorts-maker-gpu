> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (Оптимизировано для GPU)

Shorts Maker создает вертикальные видеоклипы из длинных записей геймплея. Эта библиотека Python и инструмент командной строки (CLI) обнаруживает сцены, вычисляет профили активности аудио и видео (интенсивность звука + визуальное движение) и объединяет их для ранжирования сцен по общей интенсивности. Затем инструмент выполняет кадрирование до нужного соотношения сторон и рендерит готовые к загрузке Shorts.

**Эта версия была значительно оптимизирована для видеокарт NVIDIA с использованием CUDA.**

Оригинальную версию, работающую только на CPU, можно найти здесь: [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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
  - **Аппаратное декодирование и изменение размера**: Нативная интеграция NVIDIA Video Processing Framework (VPF) через `PyNvCodec`. Декодирует, изменяет размер и преобразует цветовые пространства непосредственно на NVDEC.
  - **Обнаружение сцен**: Собственная реализация с использованием VPF и OpenCV.
  - **Анализ аудио**: Использование `torchaudio` на GPU для быстрого расчета RMS и спектрального потока.
  - **Анализ видео**: Потоковая передача в памяти GPU без копирования (zero-copy) для стабильной оценки движения (заменяет тяжелые индексы кадров).
  - **Обработка изображений**: Нативные операторы PyTorch для тяжелых операций, таких как размытие фона (сепарабельные свертки).
  - **Рендеринг**: Кастомный движок на базе PyTorch+NVENC для высокопроизводительного рендеринга (MoviePy исключен из процесса рендеринга).
  - **Надежная пакетная обработка**: Обработка видео выполняется в полностью изолированных подпроцессах, полностью очищая контексты CUDA между файлами для предотвращения фрагментации VRAM и сбоев из-за нехватки памяти (OOM) (особенно в Docker/WSL).
- Оценка активности по аудио + видео:
  - Комбинированное ранжирование с настраиваемыми весами (по умолчанию: аудио 0.6, видео 0.4).
- Сцены ранжируются по комбинированной оценке активности, а не по длительности.
- **Умная нарезка сцен**:
  - Предпочтительно выбираются полные сцены, если они укладываются в лимит времени.
  - **Дополнение сцен (Padding)**: Добавляет 1,5-секундный буфер в конец сцен для захвата завершающих анимаций и затуханий.
  - **Умная обрезка**: Для длинных сцен ищет "тихие" моменты (низкая активность аудио/движения) для обрезки, избегая резких обрывов.
- Умное кадрирование с опциональным размытием фона для невертикальных видео.
- Логика повторных попыток при рендеринге во избежание случайных сбоев.
- Настройка через переменные окружения в файле `.env`.

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

### Через PyPI (Рекомендуется)

Убедитесь, что у вас установлены драйверы NVIDIA и CUDA toolkit. Затем установите пакет напрямую:

```bash
pip install shorts-maker-gpu
```

### Ручная установка из исходного кода (Linux с CUDA)

Убедитесь, что у вас установлены драйверы NVIDIA и CUDA toolkit.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Установка библиотеки и ее зависимостей
pip install -e .
```

Если у вас возникли проблемы с тем, что PyTorch не находит GPU, обратитесь к руководству по его установке для вашей конкретной версии CUDA.

## 💡 Использование

1. Поместите исходные видео в директорию `gameplay/`.
2. Запустите инструмент командной строки (CLI):

```bash
shorts-maker process
```

При желании вы можете настроить директории ввода и вывода, а также лимиты сцен:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. Сгенерированные клипы сохраняются в директорию `generated/`.

Во время обработки в логах отображается оценка активности для каждой объединенной сцены и финальный список, отсортированный по этой оценке. Лучшие сцены (по интенсивности действий) рендерятся первыми с использованием NVENC.

## 🐳 Docker (Рекомендуется)

Самый простой способ запустить это приложение — использовать Docker с NVIDIA Container Toolkit.

**Предварительное требование**: На хосте должен быть установлен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Сборка и запуск:

*(Примечание: Если сборка прерывается с ошибкой "Segmentation fault" или ошибкой памяти, ограничьте количество ядер процессора, используя `docker build --cpuset-cpus="0,1" -t shorts-maker .`).*

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

## ⚙️ Настройка

Скопируйте файл `.env.example` в `.env` и измените значения по мере необходимости.

Поддерживаемые переменные (показаны значения по умолчанию):
- `TARGET_RATIO_W=9` — Часть ширины целевого соотношения сторон (например, 9 для 9:16).
- `TARGET_RATIO_H=16` — Часть высоты целевого соотношения сторон (например, 16 для 9:16).
- `SCENE_LIMIT=4` — Максимальное количество лучших сцен, рендерящихся для каждого исходного видео.
- `X_CENTER=0.5` — Горизонтальный центр кадрирования в диапазоне [0.0, 1.0].
- `Y_CENTER=0.5` — Вертикальный центр кадрирования в диапазоне [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Максимальная глубина повторных попыток при сбое рендеринга.
- `MIN_SHORT_LENGTH=15` — Минимальная длина шортса в секундах.
- `MAX_SHORT_LENGTH=179` — Максимальная длина шортса в секундах.
- `MAX_COMBINED_SCENE_LENGTH=300` — Максимальная комбинированная длина (в секундах).
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

Примечание: Тесты спроектированы таким образом, чтобы имитировать наличие GPU, если он отсутствует, поэтому они могут выполняться в стандартных средах CI.

## 🚑 Устранение неполадок

- **"internal compiler error: Segmentation fault" при выполнении `docker build`**: Это обычно происходит из-за нехватки памяти (OOM), когда Docker пытается скомпилировать тяжелые библиотеки C++/CUDA (такие как VPF), используя все доступные ядра процессора. Чтобы это исправить, ограничьте количество ядер процессора, используемых в процессе сборки:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(В качестве альтернативы, вы можете увеличить лимит оперативной памяти для Docker/WSL2 в настройках вашей системы).*
- **"Torch not installed" / "CUDA not available"**: Убедитесь, что вы запускаете контейнер Docker с флагом `--gpus all` или у вас локально установлен правильный набор инструментов CUDA (CUDA toolkit).
- **Ошибка NVENC**: Если `h264_nvenc` завершается с ошибкой, скрипт попытается переключиться на программное кодирование (`libx264`). Убедитесь, что ваш GPU поддерживает NVENC и что драйверы обновлены.

## 📄 Лицензия

Этот проект выпущен под [лицензией MIT](LICENSE).