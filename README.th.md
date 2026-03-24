> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งสำหรับ GPU)

Shorts Maker สร้างวิดีโอคลิปแนวตั้งจากฟุตเทจเกมเพลย์ที่มีความยาว ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก คำนวณโปรไฟล์แอ็กชันของเสียงและวิดีโอ (ความเข้มของเสียง + การเคลื่อนไหวของภาพ) และนำมารวมกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะครอบตัด (crop) ให้ได้อัตราส่วนภาพตามที่ต้องการ และเรนเดอร์เป็นวิดีโอสั้นที่พร้อมอัปโหลด

**เวอร์ชันนี้ได้รับการปรับแต่งประสิทธิภาพอย่างหนักสำหรับ NVIDIA GPU โดยใช้ CUDA**

สำหรับเวอร์ชันดั้งเดิมที่ใช้ CPU อย่างเดียว โปรดไปที่ [Shorts Maker](https://github.com/artryazanov/shorts-maker)

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

## ✨ ฟีเจอร์หลัก

- **การประมวลผลที่เร่งด้วย GPU (GPU-Accelerated Processing)**:
  - **การถอดรหัสและการปรับขนาดด้วยฮาร์ดแวร์**: ผสานรวม NVIDIA Video Processing Framework (VPF) แบบเนทีฟผ่าน `PyNvCodec` ช่วยถอดรหัส ปรับขนาด และแปลงพื้นที่สีโดยตรงบน NVDEC
  - **การตรวจจับฉาก (Scene Detection)**: อิมพลีเมนต์แบบกำหนดเองโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง (Audio Analysis)**: ใช้ `torchaudio` บน GPU เพื่อการคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **การวิเคราะห์วิดีโอ (Video Analysis)**: สตรีมหน่วยความจำ GPU แบบ Zero-copy สำหรับการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การใช้ frame indices ที่กินทรัพยากร)
  - **การประมวลผลภาพ (Image Processing)**: ใช้โอเปอเรเตอร์เนทีฟของ PyTorch สำหรับการทำงานที่กินทรัพยากรมาก เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์ (Rendering)**: เอนจินแบบกำหนดเองโดยใช้ PyTorch+NVENC สำหรับการเรนเดอร์ประสิทธิภาพสูง (ถอด MoviePy ออกจากขั้นตอนการเรนเดอร์แล้ว)
  - **การประมวลผลแบบแบตช์ที่เสถียร (Robust Batch Processing)**: การประมวลผลวิดีโอทำงานใน subprocess ที่แยกจากกันโดยสมบูรณ์ และจะเคลียร์ CUDA contexts ทั้งหมดระหว่างสลับไฟล์ เพื่อป้องกันปัญหา VRAM fragmentation และแอปเด้งจาก OOM (โดยเฉพาะใน Docker/WSL)
- การให้คะแนนแอ็กชันจากเสียง + วิดีโอ:
  - การจัดอันดับรวมที่สามารถปรับน้ำหนักได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- ฉากต่างๆ จะถูกจัดอันดับตามคะแนนแอ็กชันรวมแทนที่จะเป็นตามความยาวของวิดีโอ
- **การตัดฉากอัจฉริยะ (Smart Scene Cutting)**:
  - เลือกฉากที่สมบูรณ์ก่อนหากอยู่ในกรอบเวลาที่กำหนด
  - **Scene Padding**: เพิ่มบัฟเฟอร์ 1.5 วินาทีต่อท้ายฉากเพื่อเก็บแอนิเมชันตอนจบและการเฟด
  - **Smart Trimming**: สำหรับฉากที่ยาว จะค้นหาช่วง "เงียบ" (เสียง/การเคลื่อนไหวน้อย) เพื่อตัดหลีกเลี่ยงการจบแบบห้วนๆ
- การครอบตัดอัจฉริยะ (Smart cropping) พร้อมตัวเลือกในการเบลอพื้นหลังสำหรับฟุตเทจที่ไม่ใช่วิดีโอแนวตั้ง
- ระบบลองใหม่ (Retry logic) ระหว่างการเรนเดอร์เพื่อหลีกเลี่ยงข้อผิดพลาดที่ไม่ได้ตั้งใจ
- การกำหนดค่าผ่านตัวแปรสภาพแวดล้อมในไฟล์ `.env`

## 📋 ความต้องการของระบบ (Requirements)

- **NVIDIA GPU** ที่รองรับ CUDA
- **NVIDIA Drivers** (แนะนำให้รองรับ CUDA 13.0 ขึ้นไป)
- Python 3.12+
- FFmpeg (ใช้สำหรับแยกเสียงและเข้ารหัส NVENC)
- ไลบรารีระบบ: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีที่เกี่ยวกับวิชัน)

ไลบรารีที่จำเป็นสำหรับ Python (ดูใน `pyproject.toml`):
- `torch`, `torchaudio` (รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง NVIDIA drivers และ CUDA toolkit แล้ว จากนั้นติดตั้งแพ็กเกจได้โดยตรง:

```bash
pip install shorts-maker-gpu
```

### ติดตั้งด้วยตนเองจาก Source (Linux ที่มี CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และ CUDA toolkit เรียบร้อยแล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

หากพบปัญหาที่ PyTorch หา GPU ไม่พบ โปรดดูคู่มือการติดตั้งที่ตรงกับเวอร์ชัน CUDA ของคุณ

## 💡 วิธีการใช้งาน

1. นำวิดีโอต้นฉบับไปวางไว้ในโฟลเดอร์ `gameplay/`
2. รันเครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถกำหนดโฟลเดอร์ input, output และจำนวนฉากสูงสุด (scene limits) แบบกำหนดเองได้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างขึ้นจะถูกบันทึกไว้ในโฟลเดอร์ `generated/`

ในระหว่างการประมวลผล ล็อกจะแสดงคะแนนแอ็กชัน (action score) สำหรับแต่ละฉากรวม และจัดเรียงรายการสุดท้ายตามคะแนนนั้น ฉากอันดับต้นๆ (ตามความเข้มข้นของแอ็กชัน) จะถูกเรนเดอร์ก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**ข้อกำหนดเบื้องต้น**: ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) บนโฮสต์

การบิลด์และรัน:

*(หมายเหตุ: หากการบิลด์ขัดข้องแล้วขึ้นข้อความ "Segmentation fault" หรือมีข้อผิดพลาดเกี่ยวกับหน่วยความจำ ให้จำกัดคอร์ CPU โดยใช้คำสั่ง `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

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

โปรดสังเกตแฟล็ก `--gpus all` ซึ่งมีความจำเป็นเพื่อให้แอปพลิเคชันสามารถเข้าถึงการเร่งความเร็วด้วยฮาร์ดแวร์ได้

## ⚙️ การตั้งค่า

คัดลอก `.env.example` ไปเป็นไฟล์ `.env` และปรับค่าต่างๆ ตามที่ต้องการ

ตัวแปรที่รองรับ (แสดงพร้อมค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — อัตราส่วนความกว้างของเป้าหมาย (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — อัตราส่วนความสูงของเป้าหมาย (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนฉากสูงสุดที่จะเรนเดอร์ต่อวิดีโอต้นฉบับ 1 คลิป
- `X_CENTER=0.5` — ตำแหน่งกึ่งกลางการครอบตัดแนวนอนในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — ตำแหน่งกึ่งกลางการครอบตัดแนวตั้งในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดที่จะพยายามทำซ้ำหากการเรนเดอร์ล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาวขั้นต่ำของวิดีโอสั้น (วินาที)
- `MAX_SHORT_LENGTH=179` — ความยาวสูงสุดของวิดีโอสั้น (วินาที)
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวรวมสูงสุดของฉาก (วินาที)
- `SAVE_FFMPEG_LOGS=False` — กำหนดว่าจะบันทึกแฟ้มข้อมูลประวัติ (logs) ของ FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับการบันทึก Log (เช่น INFO, DEBUG, WARNING)

## 🛠️ สำหรับนักพัฒนา

### การทำ Linting

โปรเจกต์นี้ใช้ `ruff` สำหรับการทำ linting อย่างรวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การรัน Test

Unit tests อยู่ในโฟลเดอร์ `tests/` คุณสามารถรันด้วยคำสั่ง:

```bash
pytest -q
```

หมายเหตุ: การทดสอบถูกออกแบบมาเพื่อจำลอง (mock) การมีอยู่ของ GPU ในกรณีที่ไม่มี เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหา (Troubleshooting)

- **"internal compiler error: Segmentation fault" ในระหว่าง `docker build`**: ปัญหานี้มักจะเกิดขึ้นจากข้อผิดพลาด Out-Of-Memory (OOM) เมื่อ Docker พยายามจะคอมไพล์ไลบรารี C++/CUDA ที่ใช้ทรัพยากรสูง (เช่น VPF) โดยใช้คอร์ CPU ที่มีทั้งหมด ในการแก้ไขให้จำกัดจำนวนคอร์ CPU ที่ใช้ในขั้นตอนการบิลด์:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือคุณอาจเพิ่มขีดจำกัด RAM สำหรับ Docker/WSL2 ในการตั้งค่าระบบของคุณแทนก็ได้)*
- **"Torch not installed" / "CUDA not available"**: ตรวจสอบให้แน่ใจว่าคุณรันในคอนเทนเนอร์ Docker พร้อมแฟล็ก `--gpus all` หรือมีการติดตั้ง CUDA toolkit อย่างถูกต้องในเครื่องของคุณ
- **ข้อผิดพลาด NVENC Error**: หาก `h264_nvenc` ล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสด้วยซอฟต์แวร์ (`libx264`) ให้ตรวจสอบว่า GPU ของคุณรองรับ NVENC หรือไม่ และไดรเวอร์เป็นเวอร์ชันล่าสุดหรือไม่

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้เผยแพร่ภายใต้ [MIT License](LICENSE)