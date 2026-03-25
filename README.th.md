> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งสำหรับ GPU)

Shorts Maker เป็นเครื่องมือสำหรับสร้างคลิปวิดีโอแนวตั้งจากฟุตเทจเกมเพลย์ขนาดยาว ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก คำนวณโปรไฟล์แอ็กชันของเสียงและวิดีโอ (ความเข้มของเสียง + การเคลื่อนไหวของภาพ) และนำมารวมกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะครอบตัด (crop) ให้ได้อัตราส่วนภาพตามที่ต้องการ และเรนเดอร์เป็นคลิปสั้นที่พร้อมสำหรับอัปโหลดทันที

**เวอร์ชันนี้ได้รับการปรับแต่งอย่างหนักเพื่อใช้งานกับ NVIDIA GPU ผ่าน CUDA**

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

### [อ่านเอกสารประกอบฉบับเต็ม 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ ฟีเจอร์หลัก

- **การประมวลผลด้วย GPU (GPU-Accelerated Processing)**:
  - **ถอดรหัสและปรับขนาดด้วยฮาร์ดแวร์ (Hardware Decoding & Resizing)**: ผสานการทำงานกับ NVIDIA Video Processing Framework (VPF) ดั้งเดิมผ่าน `PyNvCodec` โดยจะถอดรหัส ปรับขนาด และแปลงสเปซสี (color spaces) โดยตรงบน NVDEC
  - **ตรวจจับฉาก (Scene Detection)**: การเขียนขึ้นเองโดยเฉพาะโดยใช้ VPF และ OpenCV
  - **วิเคราะห์เสียง (Audio Analysis)**: ใช้ `torchaudio` บน GPU เพื่อคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **วิเคราะห์วิดีโอ (Video Analysis)**: สตรีมหน่วยความจำ GPU แบบ zero-copy สำหรับการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การใช้ดัชนีเฟรมที่กินทรัพยากรสูง)
  - **ประมวลผลภาพ (Image Processing)**: ใช้ native PyTorch operators สำหรับการทำงานที่กินทรัพยากรสูง เช่น การเบลอพื้นหลัง (separable convolutions)
  - **เรนเดอร์ (Rendering)**: เอ็นจิน PyTorch+NVENC ที่เขียนขึ้นเองเพื่อการเรนเดอร์ประสิทธิภาพสูง (ถอด MoviePy ออกจากขั้นตอนการเรนเดอร์แล้ว)
  - **การประมวลผลแบบชุดที่มีความเสถียร (Robust Batch Processing)**: การประมวลผลวิดีโอทำงานในกระบวนการย่อย (subprocesses) ที่แยกจากกันอย่างสมบูรณ์ ช่วยล้าง CUDA contexts ระหว่างไฟล์ทั้งหมดเพื่อป้องกันปัญหา VRAM fragmentation และแครชแบบ OOM (โดยเฉพาะใน Docker/WSL)
- การให้คะแนนแอ็กชันของเสียง + วิดีโอ:
  - จัดอันดับแบบผสมที่สามารถปรับน้ำหนักได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- จัดอันดับฉากด้วยคะแนนแอ็กชันรวมแทนที่ความยาวของฉาก
- **การตัดฉากแบบชาญฉลาด (Smart Scene Cutting)**:
  - เลือกฉากที่สมบูรณ์ก่อน หากความยาวอยู่ในเวลาที่กำหนด
  - **ระยะขอบฉาก (Scene Padding)**: เพิ่มบัฟเฟอร์ 1.5 วินาทีในตอนท้ายของฉากเพื่อเก็บอนิเมชันช่วงออกและฉากที่ค่อยๆ เลือนหาย (fades)
  - **ตัดแต่งแบบชาญฉลาด (Smart Trimming)**: สำหรับฉากที่มีความยาวมาก จะค้นหาช่วง "เงียบ" (เสียง/การเคลื่อนไหวน้อย) ในการตัด เพื่อหลีกเลี่ยงการจบฉากแบบกะทันหัน
- การครอบตัด (cropping) แบบชาญฉลาด พร้อมตัวเลือกในการเบลอพื้นหลังสำหรับฟุตเทจที่ไม่ใช่แนวตั้ง
- ระบบลองใหม่ (Retry logic) ระหว่างการเรนเดอร์ เพื่อหลีกเลี่ยงข้อผิดพลาดแบบสุ่ม
- การกำหนดค่าผ่านตัวแปรสภาพแวดล้อมในไฟล์ `.env`

## 📋 ความต้องการของระบบ (Requirements)

- **NVIDIA GPU** ที่รองรับ CUDA
- **NVIDIA Drivers** (แนะนำเวอร์ชันที่เข้ากันได้กับ CUDA 13.0 ขึ้นไป)
- Python 3.12+
- FFmpeg (ใช้สำหรับดึงเสียงและเข้ารหัสด้วย NVENC)
- System libraries: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีทางด้านวิชัน)

ไลบรารี Python ที่ต้องใช้ (ดูเพิ่มเติมใน `pyproject.toml`):
- `torch`, `torchaudio` (ที่รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าได้ติดตั้ง NVIDIA drivers และ CUDA toolkit ไว้เรียบร้อยแล้ว จากนั้นสามารถติดตั้งแพ็กเกจได้โดยตรง:

```bash
pip install shorts-maker-gpu
```

### ติดตั้งด้วยตนเองจาก Source (Linux ที่มี CUDA)

ตรวจสอบให้แน่ใจว่าได้ติดตั้ง NVIDIA drivers และ CUDA toolkit ไว้เรียบร้อยแล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

หากคุณพบปัญหาที่ PyTorch ไม่เจอ GPU ให้ดูที่คู่มือการติดตั้ง PyTorch ให้ตรงกับเวอร์ชัน CUDA ที่คุณใช้งาน

## 💡 วิธีใช้งาน

1. นำวิดีโอต้นฉบับไปใส่ไว้ในโฟลเดอร์ `gameplay/`
2. รันเครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถปรับแต่งโฟลเดอร์ input, output และการจำกัดฉากตามที่ต้องการได้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างเสร็จแล้วจะถูกบันทึกไว้ในโฟลเดอร์ `generated/`

ในระหว่างการประมวลผล ไฟล์ log จะแสดงคะแนนแอ็กชันสำหรับแต่ละฉากที่นำมารวมกัน และรายการสุดท้ายที่เรียงลำดับตามคะแนนนั้น ฉากอันดับต้น ๆ (ตามความเข้มข้นของแอ็กชัน) จะถูกเรนเดอร์ก่อนด้วยการใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**ข้อกำหนดเบื้องต้น**: ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) บนเครื่องโฮสต์

บิลด์และรัน:

*(หมายเหตุ: หากการบิลด์ขัดข้องด้วยข้อผิดพลาด "Segmentation fault" หรือหน่วยความจำเต็ม ให้จำกัดคอร์ของ CPU โดยใช้ `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

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

โปรดสังเกตแฟล็ก `--gpus all` ซึ่งมีความจำเป็นเพื่อให้แอปพลิเคชันสามารถเข้าถึงการเร่งความเร็วทางฮาร์ดแวร์ได้

## ⚙️ การตั้งค่า (Configuration)

คัดลอกไฟล์ `.env.example` ไปเป็น `.env` และปรับเปลี่ยนค่าตามต้องการ

ตัวแปรที่รองรับ (แสดงค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — ส่วนของความกว้างสำหรับอัตราส่วนภาพที่ต้องการ (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — ส่วนของความสูงสำหรับอัตราส่วนภาพที่ต้องการ (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนฉากอันดับต้น ๆ สูงสุดที่จะทำการเรนเดอร์ต่อวิดีโอต้นฉบับ
- `SCENE_THRESHOLD=45.0` — ค่าเทรชโฮลด์ (Threshold) สำหรับการตรวจจับจุดตัดฉาก
- `X_CENTER=0.5` — จุดศูนย์กลางแนวนอนสำหรับการครอบตัด อยู่ในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — จุดศูนย์กลางแนวตั้งสำหรับการครอบตัด อยู่ในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดสำหรับการลองใหม่หากเกิดข้อผิดพลาดในการเรนเดอร์
- `MIN_SHORT_LENGTH=15` — ความยาวต่ำสุดของคลิปสั้นในหน่วยวินาที
- `MAX_SHORT_LENGTH=179` — ความยาวสูงสุดของคลิปสั้นในหน่วยวินาที
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวรวมสูงสุดของฉาก (ในหน่วยวินาที)
- `SAVE_FFMPEG_LOGS=False` — กำหนดว่าให้บันทึก log ของ FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับของการบันทึก log (เช่น INFO, DEBUG, WARNING)

## 🛠️ การพัฒนา (Development)

### การตรวจสอบโค้ด (Linting)

โปรเจกต์นี้ใช้ `ruff` เพื่อการตรวจสอบโค้ด (linting) อย่างรวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การรันชุดทดสอบ (Running Tests)

Unit tests จะอยู่ในโฟลเดอร์ `tests/` สามารถรันได้โดยใช้คำสั่ง:

```bash
pytest -q
```

หมายเหตุ: ชุดทดสอบถูกออกแบบมาให้ทำ mock การมีอยู่ของ GPU ในกรณีที่ไม่มี เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหา (Troubleshooting)

- **"internal compiler error: Segmentation fault" ระหว่าง `docker build`**: โดยทั่วไปมักเกิดจากข้อผิดพลาดหน่วยความจำไม่พอ (Out-Of-Memory หรือ OOM) เมื่อ Docker พยายามคอมไพล์ไลบรารี C++/CUDA ที่ใช้ทรัพยากรสูง (เช่น VPF) โดยใช้คอร์ CPU ทั้งหมดที่มี ในการแก้ไขให้จำกัดจำนวนคอร์ CPU ที่ใช้ในระหว่างขั้นตอนการบิลด์:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือคุณอาจไปเพิ่มขีดจำกัด RAM สำหรับ Docker/WSL2 ได้ในการตั้งค่าระบบของคุณ)*
- **"WSL integration with distro unexpectedly stopped" / OOM ระหว่าง `docker run`**: การประมวลผลวิดีโอที่มีความละเอียดสูงอาจใช้ RAM/VRAM ในปริมาณมาก ซึ่งเป็นสาเหตุให้เครื่องเสมือน WSL2 แครชจากข้อผิดพลาดหน่วยความจำไม่พอ (OOM) ในการแก้ไข ให้จำกัดจำนวนคอร์ CPU ที่คอนเทนเนอร์ใช้งานได้ในระหว่างการรันโดยการเพิ่มแฟล็ก `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: ตรวจสอบให้แน่ใจว่าคุณกำลังรันอยู่ในคอนเทนเนอร์ Docker พร้อมแฟล็ก `--gpus all` หรือติดตั้ง CUDA toolkit ที่ถูกต้องเอาไว้ในเครื่องแล้ว
- **NVENC Error**: หาก `h264_nvenc` ทำงานล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสด้วยซอฟต์แวร์ (`libx264`) แทน โปรดตรวจสอบว่า GPU ของคุณรองรับ NVENC หรือไม่ และไดรเวอร์ได้รับการอัปเดตแล้วหรือยัง

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้เผยแพร่ภายใต้ [MIT License](LICENSE)