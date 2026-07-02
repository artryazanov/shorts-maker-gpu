> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งเพื่อการใช้งานกับ GPU)

Shorts Maker สร้างวิดีโอคลิปแนวตั้งจากฟุตเทจเกมเพลย์ที่ยาวกว่า ไลบรารี Python และเครื่องมือ CLI นี้สามารถตรวจจับฉาก (scenes) คำนวณโปรไฟล์แอ็คชันของเสียงและวิดีโอ (ความเข้มของเสียง + การเคลื่อนไหวของภาพ) และนำมารวมกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะทำการครอบตัด (crop) ให้ได้อัตราส่วนที่ต้องการ และเรนเดอร์เป็นคลิป Shorts ที่พร้อมสำหรับการอัปโหลด

**เวอร์ชันนี้ได้รับการปรับให้เหมาะสมอย่างหนักสำหรับการใช้งานกับ GPU NVIDIA โดยใช้ CUDA**

สำหรับเวอร์ชันเดิมที่ใช้เฉพาะ CPU กรุณาเข้าไปที่ [Shorts Maker](https://github.com/artryazanov/shorts-maker)

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

- **การประมวลผลที่เร่งความเร็วด้วย GPU (GPU-Accelerated Processing)**:
  - **การถอดรหัสและการปรับขนาดด้วยฮาร์ดแวร์ (Hardware Decoding & Resizing)**: ผสานการทำงานกับ NVIDIA Video Processing Framework (VPF) แบบเนทีฟผ่าน `PyNvCodec` ถอดรหัส ปรับขนาด และแปลงปริภูมิสีโดยตรงบน NVDEC
  - **การตรวจจับฉาก (Scene Detection)**: อิมพลีเมนต์แบบกำหนดเองโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง (Audio Analysis)**: ใช้ `torchaudio` บน GPU เพื่อการคำนวณ RMS และ spectral flux ที่รวดเร็ว
  - **การวิเคราะห์วิดีโอ (Video Analysis)**: การสตรีมหน่วยความจำ GPU แบบ zero-copy สำหรับการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การดึงดัชนีเฟรมที่กินทรัพยากร)
  - **การประมวลผลภาพ (Image Processing)**: ใช้ตัวดำเนินการของ PyTorch แบบเนทีฟสำหรับการทำงานหนักๆ เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์ (Rendering)**: เอนจินแบบกำหนดเอง (PyTorch+NVENC) สำหรับการเรนเดอร์ประสิทธิภาพสูง (นำ MoviePy ออกจากขั้นตอนการเรนเดอร์)
  - **การประมวลผลแบบแบตช์ที่มีประสิทธิภาพ (Robust Batch Processing)**: การประมวลผลวิดีโอจะทำงานในกระบวนการย่อยที่แยกออกจากกันอย่างสมบูรณ์ ล้างบริบทของ CUDA ระหว่างไฟล์ทั้งหมดเพื่อป้องกันการกระจัดกระจายของ VRAM และการเกิดปัญหา OOM (โดยเฉพาะใน Docker/WSL)
  - **การจัดการ VFR ที่แม่นยำ (Accurate VFR Handling)**: ดึง Presentation Timestamps (PTS) ที่แท้จริงออกจากแพ็กเก็ตวิดีโอโดยตรง เพื่อป้องกันไม่ให้เสียงและวิดีโอไม่ตรงกัน จัดการกับวิดีโอแบบ Variable Frame Rate (VFR) ได้อย่างไร้รอยต่อ
- การให้คะแนนแอ็คชันทั้งเสียงและวิดีโอ:
  - จัดอันดับรวมด้วยน้ำหนักที่สามารถปรับได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- ฉากถูกจัดอันดับจากคะแนนแอ็คชันรวมมากกว่าระยะเวลาของฉาก
- **การตัดฉากแบบอัจฉริยะ (Smart Scene Cutting)**:
  - ให้ความสำคัญกับฉากที่สมบูรณ์หากฉากเหล่านั้นอยู่ภายในระยะเวลาที่กำหนด
  - **การเพิ่มระยะเวลาให้ฉาก (Scene Padding)**: เพิ่มบัฟเฟอร์ 1.5 วินาทีต่อท้ายฉากเพื่อจับภาพแอนิเมชันตอนจบและเอฟเฟกต์เฟด
  - **การตัดแต่งแบบอัจฉริยะ (Smart Trimming)**: สำหรับฉากที่ยาวเกินไป ระบบจะค้นหาช่วงเวลาที่ "เงียบ" (เสียง/การเคลื่อนไหวน้อย) ในการตัด เพื่อหลีกเลี่ยงการตัดแบบกะทันหัน
- การครอบตัดแบบอัจฉริยะพร้อมตัวเลือกการเบลอพื้นหลังสำหรับฟุตเทจที่ไม่ใช่วิดีโอแนวตั้ง
- ตรรกะการลองใหม่ (Retry) ระหว่างการเรนเดอร์เพื่อหลีกเลี่ยงข้อผิดพลาดที่เกิดขึ้นแบบสุ่ม
- การกำหนดค่าผ่านตัวแปรสภาพแวดล้อม (environment variables) ในไฟล์ `.env`

## 📋 ความต้องการของระบบ

- **GPU NVIDIA** ที่รองรับ CUDA
- **ไดรเวอร์ NVIDIA** (แนะนำรุ่นที่เข้ากันได้กับ CUDA 13.0+)
- Python 3.12+
- FFmpeg (ใช้สำหรับดึงเสียงและการเข้ารหัสด้วย NVENC)
- ไลบรารีระบบ: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีประเภท vision)

แพ็กเกจที่จำเป็นของ Python (ดูที่ `pyproject.toml`):
- `torch`, `torchaudio` (ที่รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และชุดเครื่องมือ CUDA เรียบร้อยแล้ว จากนั้นติดตั้งแพ็กเกจได้โดยตรง:

```bash
pip install shorts-maker-gpu
```

### การติดตั้งด้วยตนเองจาก Source (Linux ที่มี CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และชุดเครื่องมือ CUDA เรียบร้อยแล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# ติดตั้งไลบรารีและ dependencies
pip install -e .
```

หากคุณพบปัญหาที่ PyTorch ไม่พบ GPU ให้อ้างอิงตามคู่มือการติดตั้งของ PyTorch สำหรับเวอร์ชัน CUDA โดยเฉพาะของคุณ

## 💡 วิธีการใช้งาน

1. นำวิดีโอต้นฉบับไปไว้ในไดเรกทอรี `gameplay/`
2. เรียกใช้เครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถกำหนดไดเรกทอรีสำหรับนำเข้า ส่งออก และจำกัดจำนวนฉากเองได้ด้วยคำสั่งนี้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างขึ้นจะถูกบันทึกไว้ในไดเรกทอรี `generated/`

ในระหว่างการประมวลผล บันทึก (log) จะแสดงคะแนนแอ็คชันสำหรับแต่ละฉากที่รวมกัน และจะแสดงรายการสุดท้ายที่จัดเรียงตามคะแนนนั้น ฉากอันดับต้น ๆ (ตามความเข้มข้นของแอ็คชัน) จะถูกเรนเดอร์ก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**ข้อกำหนดเบื้องต้น**: จะต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) ลงบนเครื่องโฮสต์

คำสั่งบิลด์และรัน:

*(หมายเหตุ: หากการบิลด์ล้มเหลวพร้อมข้อความ "Segmentation fault" หรือมีข้อผิดพลาดเกี่ยวกับหน่วยความจำ ให้ทำการจำกัดคอร์ประมวลผล CPU โดยใช้คำสั่ง `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

```bash
docker build -t shorts-maker .

# รันด้วยการอนุญาตให้เข้าถึง GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

โปรดสังเกตที่แฟล็ก `--gpus all` ซึ่งมีความสำคัญอย่างยิ่งสำหรับแอปพลิเคชัน เพื่อให้สามารถเข้าถึงการเร่งความเร็วด้วยฮาร์ดแวร์ได้

## ⚙️ การตั้งค่า

คัดลอกไฟล์ `.env.example` ไปเป็น `.env` และปรับเปลี่ยนค่าตามความต้องการ

ตัวแปรที่รองรับ (ค่าที่แสดงคือค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — ส่วนความกว้างของอัตราส่วนภาพเป้าหมาย (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — ส่วนความสูงของอัตราส่วนภาพเป้าหมาย (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนสูงสุดของฉากระดับท็อปที่จะถูกเรนเดอร์ต่อหนึ่งวิดีโอต้นฉบับ
- `SCENE_THRESHOLD=45.0` — ค่าเกณฑ์สำหรับการพิจารณาการตัดฉาก
- `X_CENTER=0.5` — จุดศูนย์กลางการครอบตัดแนวนอนในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — จุดศูนย์กลางการครอบตัดแนวตั้งในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดในการลองใหม่ หากการเรนเดอร์ล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาวขั้นต่ำของคลิปชอร์ตในหน่วยวินาที
- `MAX_SHORT_LENGTH=179` — ความยาวสูงสุดของคลิปชอร์ตในหน่วยวินาที
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวสูงสุดเมื่อรวมฉาก (ในหน่วยวินาที)
- `SAVE_FFMPEG_LOGS=False` — เลือกกำหนดว่าจะบันทึก log ของ FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับการบันทึก log (เช่น INFO, DEBUG, WARNING)

## 🛠️ การพัฒนา

### การทำ Linting

โปรเจกต์นี้ใช้ `ruff` เพื่อตรวจสอบคุณภาพโค้ด (Linting) อย่างรวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การทดสอบ

Unit tests ถูกเก็บไว้ในโฟลเดอร์ `tests/` สั่งรันการทดสอบด้วยคำสั่ง:

```bash
pytest -q
```

หมายเหตุ: การทดสอบถูกออกแบบมาเพื่อจำลอง (mock) สถานะของ GPU ในกรณีที่เครื่องไม่มี GPU เพื่อให้สามารถทดสอบทำงานในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหาเบื้องต้น (Troubleshooting)

- **มีข้อความ "internal compiler error: Segmentation fault" ในระหว่างใช้คำสั่ง `docker build`**: ปัญหานี้มักเกิดจากหน่วยความจำไม่เพียงพอ (Out-Of-Memory / OOM) เมื่อ Docker พยายามจะคอมไพล์ไลบรารี C++/CUDA ที่ใช้ทรัพยากรสูง (เช่น VPF) โดยใช้คอร์ประมวลผล CPU ที่มีทั้งหมด หากต้องการแก้ไข ให้จำกัดจำนวนคอร์ประมวลผล CPU ที่ใช้ระหว่างขั้นตอนการบิลด์:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือในอีกทางเลือก คุณสามารถไปเพิ่มขีดจำกัด RAM สำหรับ Docker/WSL2 ได้ในการตั้งค่าของระบบคุณเอง)*
- **มีข้อความ "WSL integration with distro unexpectedly stopped" / ปัญหา OOM ระหว่างใช้คำสั่ง `docker run`**: การประมวลผลวิดีโอความละเอียดสูงอาจใช้ RAM/VRAM จำนวนมาก ทำให้ Virtual machine (VM) ของเครื่อง WSL2 ล่ม เนื่องจากข้อผิดพลาดหน่วยความจำไม่เพียงพอ (OOM) หากต้องการแก้ปัญหานี้ ให้จำกัดจำนวนคอร์ประมวลผล CPU ที่คอนเทนเนอร์สามารถใช้งานได้ระหว่างการรัน โดยใส่แฟล็ก `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: ตรวจสอบให้แน่ใจว่าคุณกำลังรันใน Docker container โดยระบุแฟล็ก `--gpus all` เรียบร้อยแล้ว หรือมีเครื่องมือ CUDA toolkit ที่ถูกต้องติดตั้งอยู่ในเครื่องแล้วหรือไม่
- **ข้อผิดพลาดจาก NVENC**: หากการทำงานด้วย `h264_nvenc` ล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสด้วยซอฟต์แวร์แทน (`libx264`) โปรดตรวจสอบว่า GPU ของคุณรองรับการใช้งาน NVENC หรือไม่ และคุณได้ทำการอัปเดตไดรเวอร์แล้วหรือยัง

## 📄 ลิขสิทธิ์ (License)

โปรเจกต์นี้เผยแพร่ภายใต้ลิขสิทธิ์ [MIT License](LICENSE)