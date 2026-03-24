> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งสำหรับ GPU)

Shorts Maker ใช้สำหรับสร้างวิดีโอแนวตั้งจากคลิปวิดีโอเกมเพลย์ที่ยาวกว่า ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก (scenes) คำนวณโปรไฟล์แอ็กชันของเสียงและวิดีโอ (ความดังของเสียง + การเคลื่อนไหวของภาพ) และนำมารวมกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะครอปให้ได้อัตราส่วนที่ต้องการและเรนเดอร์เป็นวิดีโอสั้น (shorts) ที่พร้อมอัปโหลดทันที

**เวอร์ชันนี้ได้รับการปรับแต่งอย่างเต็มรูปแบบสำหรับ NVIDIA GPU โดยใช้ CUDA**

สำหรับเวอร์ชันดั้งเดิมที่ใช้ CPU เท่านั้น โปรดไปที่ [Shorts Maker](https://github.com/artryazanov/shorts-maker)

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

### [อ่านเอกสารฉบับเต็ม 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ คุณสมบัติ

- **ประมวลผลด้วยความเร็วจาก GPU (GPU-Accelerated Processing)**:
  - **ถอดรหัสและปรับขนาดภาพด้วยฮาร์ดแวร์ (Hardware Decoding & Resizing)**: ผสานการทำงานกับ NVIDIA Video Processing Framework (VPF) ผ่าน `PyNvCodec` โดยจะถอดรหัส ปรับขนาด และแปลงปริภูมิสีโดยตรงบน NVDEC
  - **การตรวจจับฉาก (Scene Detection)**: อิมพลีเมนต์ขึ้นเป็นพิเศษโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง (Audio Analysis)**: ใช้ `torchaudio` บน GPU เพื่อการคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **การวิเคราะห์วิดีโอ (Video Analysis)**: สตรีมหน่วยความจำ GPU แบบ Zero-copy เพื่อการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การใช้ index ของเฟรมที่กินทรัพยากรสูง)
  - **การประมวลผลภาพ (Image Processing)**: ใช้ PyTorch operators สำหรับการคำนวณที่หนักหน่วง เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์ (Rendering)**: เอนจินแบบกำหนดเองโดยใช้ PyTorch+NVENC เพื่อการเรนเดอร์ประสิทธิภาพสูง (ถอด MoviePy ออกจากกระบวนการเรนเดอร์แล้ว)
  - **การประมวลผลแบบแบตช์ที่เสถียร (Robust Batch Processing)**: การประมวลผลวิดีโอจะทำงานในโปรเซสย่อยที่แยกออกจากกันโดยสมบูรณ์ และจะล้าง CUDA contexts ทั้งหมดระหว่างไฟล์เพื่อป้องกันการแตกกระจายของ VRAM และปัญหา OOM (Out Of Memory) แครช (โดยเฉพาะใน Docker/WSL)
- การให้คะแนนแอ็กชันจากเสียง + วิดีโอ:
  - การจัดอันดับแบบผสมผสานพร้อมน้ำหนักที่ปรับแต่งได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- จัดอันดับฉากโดยอิงจากคะแนนแอ็กชันรวมแทนระยะเวลาความยาว
- **การตัดฉากอัจฉริยะ (Smart Scene Cutting)**:
  - เลือกฉากแบบเต็มฉากเป็นหลัก หากอยู่ในขอบเขตเวลาที่กำหนด
  - **เพิ่มเวลาเผื่อสำหรับฉาก (Scene Padding)**: เพิ่มเวลาช่วงบัฟเฟอร์ 1.5 วินาทีต่อท้ายฉากเพื่อเก็บภาพแอนิเมชันช่วงจบและการเฟด (Fades)
  - **การตัดอย่างอัจฉริยะ (Smart Trimming)**: สำหรับฉากที่ยาวเกินไป ระบบจะค้นหาจังหวะที่ "เงียบ" (เสียง/การเคลื่อนไหวน้อย) เพื่อตัดวิดีโอ ช่วยหลีกเลี่ยงการตัดจบแบบกะทันหัน
- การครอปอัจฉริยะ พร้อมตัวเลือกภาพพื้นหลังเบลอสำหรับวิดีโอที่ไม่ใช่แนวตั้ง
- ระบบจำลองการทำซ้ำ (Retry logic) ระหว่างการเรนเดอร์เพื่อป้องกันข้อผิดพลาดที่ไม่คาดคิด
- ตั้งค่าระบบผ่านตัวแปรสภาพแวดล้อม (Environment variables) ในไฟล์ `.env`

## 📋 ความต้องการของระบบ

- **NVIDIA GPU** ที่รองรับ CUDA
- **NVIDIA Drivers** (แนะนำเวอร์ชันที่รองรับ CUDA 13.0+)
- Python 3.12+
- FFmpeg (ใช้สำหรับดึงเสียงและเข้ารหัสด้วย NVENC)
- System libraries: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีที่จัดการเกี่ยวกับภาพ/วิดีโอ)

Python dependencies (ดูได้ใน `pyproject.toml`):
- `torch`, `torchaudio` (ที่รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง NVIDIA drivers และ CUDA toolkit ไว้เรียบร้อยแล้ว จากนั้นติดตั้งแพ็กเกจได้โดยตรง:

```bash
pip install shorts-maker-gpu
```

### ตั้งค่าแบบ Manual จาก Source Code (Linux พร้อม CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง NVIDIA drivers และ CUDA toolkit ไว้เรียบร้อยแล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

หากคุณพบปัญหา PyTorch หา GPU ไม่เจอ โปรดอ้างอิงคู่มือการติดตั้ง PyTorch สำหรับเวอร์ชัน CUDA ที่คุณใช้งาน

## 💡 การใช้งาน

1. วางไฟล์วิดีโอต้นฉบับลงในไดเรกทอรี `gameplay/`
2. รันเครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถเลือกปรับแต่งไดเรกทอรี input, output และจำนวนฉากสูงสุดที่ต้องการได้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปวิดีโอที่ถูกสร้างขึ้นจะถูกบันทึกไว้ในไดเรกทอรี `generated/`

ระหว่างการประมวลผล ไฟล์ล็อกจะแสดงคะแนนแอ็กชันสำหรับแต่ละฉากที่รวมไว้ และรายชื่อสุดท้ายจะถูกจัดเรียงตามคะแนนดังกล่าว ฉากที่ดีที่สุด (วัดจากความเข้มข้นของแอ็กชัน) จะถูกเรนเดอร์ออกมาก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**สิ่งที่จำเป็นต้องมี**: ต้องทำการติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) บนเครื่องโฮสต์

ขั้นตอน Build และรัน:

*(หมายเหตุ: หากการ Build ล้มเหลวพร้อมข้อผิดพลาด "Segmentation fault" หรือ memory error ให้จำกัดจำนวนคอร์ CPU โดยใช้ `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

```bash
docker build -t shorts-maker .

# Run with GPU access
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

สังเกตที่แฟล็ก `--gpus all` ซึ่งมีความสำคัญมาก เพื่อให้แอปพลิเคชันสามารถเข้าถึงการเร่งความเร็วด้วยฮาร์ดแวร์ได้

## ⚙️ การตั้งค่า

คัดลอกไฟล์ `.env.example` ไปเป็น `.env` และปรับเปลี่ยนค่าตามความต้องการ

ตัวแปรที่รองรับ (พร้อมค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — สัดส่วนความกว้างของอัตราส่วนภาพที่ต้องการ (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — สัดส่วนความสูงของอัตราส่วนภาพที่ต้องการ (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนสูงสุดของฉากที่ดีที่สุดที่จะถูกเรนเดอร์ต่อ 1 วิดีโอต้นฉบับ
- `SCENE_THRESHOLD=45.0` — เกณฑ์ขั้นต่ำสำหรับการตัดตรวจจับฉาก (Scene detection cuts)
- `X_CENTER=0.5` — จุดศูนย์กลางในแนวนอนสำหรับการครอป (อยู่ในช่วง [0.0, 1.0])
- `Y_CENTER=0.5` — จุดศูนย์กลางในแนวตั้งสำหรับการครอป (อยู่ในช่วง [0.0, 1.0])
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดสำหรับการลองใหม่ หากการเรนเดอร์ล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาวขั้นต่ำของวิดีโอสั้น (หน่วยเป็นวินาที)
- `MAX_SHORT_LENGTH=179` — ความยาวสูงสุดของวิดีโอสั้น (หน่วยเป็นวินาที)
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวสูงสุดของฉากที่นำมารวมกัน (หน่วยเป็นวินาที)
- `SAVE_FFMPEG_LOGS=False` — กำหนดว่าจะบันทึกไฟล์ล็อก FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับการบันทึกล็อก (เช่น INFO, DEBUG, WARNING)

## 🛠️ การพัฒนา (Development)

### การตรวจสอบโค้ด (Linting)

โปรเจกต์นี้ใช้ `ruff` เพื่อการตรวจสอบโค้ดที่รวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การรันการทดสอบ

Unit tests จะอยู่ในโฟลเดอร์ `tests/` สามารถรันได้ด้วยคำสั่ง:

```bash
pytest -q
```

หมายเหตุ: การทดสอบถูกออกแบบมาให้จำลองการมีอยู่ของ GPU หากระบบไม่มี เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหาเบื้องต้น

- **"internal compiler error: Segmentation fault" ระหว่าง `docker build`**: โดยทั่วไปมักเกิดจากข้อผิดพลาด Out-Of-Memory (OOM) เมื่อ Docker พยายามคอมไพล์ไลบรารี C++/CUDA ที่หนักๆ (เช่น VPF) โดยใช้คอร์ CPU ที่มีทั้งหมด หากต้องการแก้ไขปัญหานี้ ให้จำกัดจำนวนคอร์ CPU ที่ใช้ระหว่างขั้นตอนการ Build:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือสามารถเลือกเพิ่มลิมิตของ RAM สำหรับ Docker/WSL2 ในการตั้งค่าระบบของคุณได้เช่นกัน)*
- **"Torch not installed" / "CUDA not available"**: ตรวจสอบให้แน่ใจว่าคุณกำลังรันภายใน Docker container โดยใช้คำสั่ง `--gpus all` หรือได้ติดตั้ง CUDA toolkit เวอร์ชันที่ถูกต้องไว้บนเครื่องของคุณแล้ว
- **NVENC Error**: หาก `h264_nvenc` ล้มเหลว สคริปต์จะพยายามใช้ทางเลือกสำรองเป็นการเข้ารหัสด้วยซอฟต์แวร์แทน (`libx264`) โปรดตรวจสอบว่า GPU ของคุณรองรับ NVENC หรือไม่ และเช็กว่าไดรเวอร์อัปเดตล่าสุดหรือยัง

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้เผยแพร่ภายใต้เงื่อนไขของ [MIT License](LICENSE)