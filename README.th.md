> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (ปรับปรุงประสิทธิภาพสำหรับ GPU)

Shorts Maker สร้างคลิปวิดีโอแนวตั้งจากฟุตเทจเกมเพลย์ที่มีความยาว ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก คำนวณโปรไฟล์แอ็คชันของเสียงและวิดีโอ (ความดังของเสียง + การเคลื่อนไหวของภาพ) และรวมเข้าด้วยกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะครอบตัด (crop) ให้ได้อัตราส่วนภาพที่ต้องการ และเรนเดอร์เป็นวิดีโอสั้น (shorts) ที่พร้อมอัปโหลด

**เวอร์ชันนี้ได้รับการปรับปรุงประสิทธิภาพอย่างมากสำหรับ NVIDIA GPU โดยใช้ CUDA**

สำหรับเวอร์ชันดั้งเดิมที่ใช้เฉพาะ CPU โปรดไปที่ [Shorts Maker](https://github.com/artryazanov/shorts-maker)

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

## ✨ คุณสมบัติ

- **การประมวลผลด้วยการเร่งความเร็ว GPU**:
  - **การถอดรหัสฮาร์ดแวร์และการปรับขนาด**: การผสานการทำงานกับ NVIDIA Video Processing Framework (VPF) แบบเนทีฟผ่าน `PyNvCodec` โดยจะถอดรหัส ปรับขนาด และแปลงปริภูมิสี (color spaces) โดยตรงบน NVDEC
  - **การตรวจจับฉาก**: การนำไปใช้งานแบบปรับแต่งเองโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง**: ใช้ `torchaudio` บน GPU เพื่อการคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **การวิเคราะห์วิดีโอ**: การสตรีมหน่วยความจำ GPU แบบ zero-copy สำหรับการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การใช้ index เฟรมที่กินทรัพยากรมาก)
  - **การประมวลผลภาพ**: ใช้โอเปอเรเตอร์ PyTorch แบบเนทีฟสำหรับการทำงานที่หนักหน่วง เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์**: เอนจิน PyTorch+NVENC แบบปรับแต่งเองเพื่อการเรนเดอร์ประสิทธิภาพสูง (นำ MoviePy ออกจากขั้นตอนการเรนเดอร์แล้ว)
  - **การประมวลผลเป็นชุดที่เสถียร**: การประมวลผลวิดีโอจะทำงานในโปรเซสย่อยที่แยกออกจากกันอย่างสมบูรณ์ โดยจะล้างบริบท (contexts) ของ CUDA ระหว่างไฟล์ทั้งหมดเพื่อป้องกันการแตกกระจายของ VRAM และปัญหา OOM (หน่วยความจำไม่พอ) (โดยเฉพาะใน Docker/WSL)
- การให้คะแนนแอ็คชันด้วยเสียง + วิดีโอ:
  - การจัดอันดับแบบรวมด้วยน้ำหนักที่ปรับได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- ฉากต่างๆ จะถูกจัดอันดับตามคะแนนแอ็คชันรวมแทนที่จะเป็นระยะเวลา
- **การตัดฉากอัจฉริยะ**:
  - เลือกรักษาฉากที่สมบูรณ์ไว้หากความยาวอยู่ในขีดจำกัดเวลา
  - **การเพิ่มระยะเวลาของฉาก**: เพิ่มบัฟเฟอร์ 1.5 วินาทีที่ตอนท้ายของฉากเพื่อจับภาพแอนิเมชันขาออกและการเฟด
  - **การตัดขอบอัจฉริยะ**: สำหรับฉากที่ยาว จะค้นหาช่วงเวลาที่ "เงียบ" (เสียง/การเคลื่อนไหวน้อย) ในการตัด เพื่อหลีกเลี่ยงการจบแบบกะทันหัน
- การครอบตัดอัจฉริยะพร้อมตัวเลือกพื้นหลังเบลอสำหรับฟุตเทจที่ไม่ใช่วิดีโอแนวตั้ง
- ระบบการลองใหม่ (Retry) ระหว่างการเรนเดอร์เพื่อหลีกเลี่ยงความล้มเหลวที่ผิดพลาด
- การกำหนดค่าผ่านตัวแปรสภาพแวดล้อม `.env`

## 📋 ความต้องการของระบบ

- **NVIDIA GPU** ที่รองรับ CUDA
- **ไดรเวอร์ NVIDIA** (แนะนำรุ่นที่เข้ากันได้กับ CUDA 13.0+)
- Python 3.12+
- FFmpeg (ใช้สำหรับการแยกเสียงและการเข้ารหัส NVENC)
- ไลบรารีระบบ: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีการมองเห็น)

ไลบรารีที่จำเป็นสำหรับ Python (ดูใน `pyproject.toml`):
- `torch`, `torchaudio` (รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และ CUDA toolkit แล้ว จากนั้นติดตั้งแพ็กเกจโดยตรง:

```bash
pip install shorts-maker-gpu
```

### ตั้งค่าด้วยตนเองจากซอร์สโค้ด (Linux พร้อม CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และ CUDA toolkit แล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# ติดตั้งไลบรารีและอ้างอิงที่จำเป็น
pip install -e .
```

หากคุณพบปัญหาเกี่ยวกับ PyTorch ที่ไม่พบ GPU โปรดดูคู่มือการติดตั้งสำหรับเวอร์ชัน CUDA เฉพาะของคุณ

## 💡 การใช้งาน

1. วางวิดีโอต้นฉบับไว้ในโฟลเดอร์ `gameplay/`
2. รันเครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถเลือกกำหนดโฟลเดอร์อินพุตและเอาต์พุต รวมถึงขีดจำกัดของฉากเองได้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างเสร็จจะถูกบันทึกไว้ในโฟลเดอร์ `generated/`

ในระหว่างการประมวลผล บันทึก (log) จะแสดงคะแนนแอ็คชันของแต่ละฉากรวม และรายการสุดท้ายที่จัดเรียงตามคะแนนนั้น ฉากอันดับต้นๆ (ตามความเข้มข้นของแอ็คชัน) จะถูกเรนเดอร์ก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**สิ่งสำคัญก่อนเริ่ม**: ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) บนโฮสต์

สร้างอิมเมจและรัน:

*(หมายเหตุ: หากกระบวนการบิลด์ล้มเหลวโดยแจ้งข้อผิดพลาด "Segmentation fault" หรือหน่วยความจำไม่เพียงพอ ให้จำกัดแกน CPU โดยใช้ `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

```bash
docker build -t shorts-maker .

# รันโดยอนุญาตให้เข้าถึง GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

โปรดสังเกตแฟล็ก `--gpus all` ซึ่งมีความสำคัญเพื่อให้แอปพลิเคชันสามารถเข้าถึงการเร่งความเร็วของฮาร์ดแวร์ได้

## ⚙️ การตั้งค่าคอนฟิก

คัดลอกไฟล์ `.env.example` เป็น `.env` และปรับค่าต่างๆ ตามต้องการ

ตัวแปรที่รองรับ (ค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — ส่วนของความกว้างสำหรับอัตราส่วนภาพเป้าหมาย (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — ส่วนของความสูงสำหรับอัตราส่วนภาพเป้าหมาย (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนฉากที่ดีที่สุดสูงสุดที่จะเรนเดอร์ต่อวิดีโอต้นฉบับหนึ่งไฟล์
- `X_CENTER=0.5` — จุดศูนย์กลางการครอบตัดแนวนอนในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — จุดศูนย์กลางการครอบตัดแนวตั้งในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดในการลองเรนเดอร์ซ้ำหากเกิดความล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาวขั้นต่ำของวิดีโอสั้นในหน่วยวินาที
- `MAX_SHORT_LENGTH=179` — ความยาวสูงสุดของวิดีโอสั้นในหน่วยวินาที
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวรวมสูงสุด (ในหน่วยวินาที)
- `SAVE_FFMPEG_LOGS=False` — เลือกว่าจะบันทึก log ของ FFmpeg ระหว่างการเรนเดอร์หรือไม่

## 🛠️ การพัฒนา

### การทำ Linting

โปรเจกต์นี้ใช้ `ruff` สำหรับการตรวจสอบโค้ดอย่างรวดเร็ว (linting)

```bash
pip install ruff
ruff check .
```

## 🧪 การรันชุดทดสอบ

Unit tests จะอยู่ในโฟลเดอร์ `tests/` สามารถรันได้โดยใช้:

```bash
pytest -q
```

หมายเหตุ: ชุดทดสอบถูกออกแบบมาเพื่อจำลอง (mock) สถานะของ GPU ในกรณีที่ไม่มีอุปกรณ์ เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหา

- **"internal compiler error: Segmentation fault" ระหว่างรัน `docker build`**: ข้อผิดพลาดนี้มักเกิดจากหน่วยความจำไม่เพียงพอ (Out-Of-Memory หรือ OOM) เมื่อ Docker พยายามคอมไพล์ไลบรารี C++/CUDA ที่ใช้งานหนัก (เช่น VPF) โดยใช้แกนประมวลผล CPU ทั้งหมดที่มี เพื่อแก้ไขปัญหานี้ ให้จำกัดจำนวนแกน CPU ที่ใช้งานระหว่างกระบวนการบิลด์:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรืออีกวิธีหนึ่ง คุณสามารถเพิ่มขีดจำกัด RAM สำหรับ Docker/WSL2 ในการตั้งค่าระบบของคุณได้)*
- **"Torch not installed" / "CUDA not available"**: ตรวจสอบให้แน่ใจว่าคุณกำลังรันภายในคอนเทนเนอร์ Docker พร้อมแฟล็ก `--gpus all` หรือได้ติดตั้ง CUDA toolkit เวอร์ชันที่ถูกต้องไว้ในเครื่องของคุณแล้ว
- **ข้อผิดพลาด NVENC**: หาก `h264_nvenc` ล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสด้วยซอฟต์แวร์ (`libx264`) ตรวจสอบว่า GPU ของคุณรองรับ NVENC หรือไม่ และไดรเวอร์เป็นเวอร์ชันล่าสุดหรือยัง

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้เผยแพร่ภายใต้ [MIT License](LICENSE)