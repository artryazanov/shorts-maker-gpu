> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งสำหรับ GPU)

Shorts Maker ใช้สำหรับสร้างคลิปวิดีโอแนวตั้งจากฟุตเทจเกมเพลย์ที่มีความยาว ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก คำนวณโปรไฟล์แอ็กชันของเสียงและวิดีโอ (ความเข้มข้นของเสียง + การเคลื่อนไหวของภาพ) และรวมเข้าด้วยกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นจะทำการครอบตัดให้อยู่ในอัตราส่วนภาพที่ต้องการ และเรนเดอร์ออกมาเป็นวิดีโอสั้นที่พร้อมอัปโหลดในทันที

**เวอร์ชันนี้ได้รับการปรับแต่งอย่างเต็มรูปแบบสำหรับ NVIDIA GPU โดยใช้ CUDA**

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

## ✨ ฟีเจอร์หลัก (Features)

- **การประมวลผลที่เร่งความเร็วด้วย GPU (GPU-Accelerated Processing)**:
  - **การถอดรหัสและการปรับขนาดด้วยฮาร์ดแวร์ (Hardware Decoding & Resizing)**: ผสานการทำงานกับ NVIDIA Video Processing Framework (VPF) ดั้งเดิมผ่าน `PyNvCodec` โดยจะทำการถอดรหัส ปรับขนาด และแปลงปริภูมิสี (color spaces) บน NVDEC โดยตรง
  - **การตรวจจับฉาก (Scene Detection)**: การนำไปใช้งานแบบกำหนดเองโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง (Audio Analysis)**: ใช้ `torchaudio` บน GPU เพื่อคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **การวิเคราะห์วิดีโอ (Video Analysis)**: การสตรีมหน่วยความจำ GPU แบบ Zero-copy เพื่อการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การใช้ frame indices ที่กินทรัพยากรสูง)
  - **การประมวลผลภาพ (Image Processing)**: ใช้ตัวดำเนินการดั้งเดิมของ PyTorch สำหรับการทำงานที่กินทรัพยากรสูง เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์ (Rendering)**: เอนจิน PyTorch+NVENC แบบกำหนดเองเพื่อการเรนเดอร์ที่มีประสิทธิภาพสูง (นำ MoviePy ออกจากเส้นทางการเรนเดอร์แล้ว)
  - **การประมวลผลแบบแบตช์ที่เสถียร (Robust Batch Processing)**: การประมวลผลวิดีโอทำงานในกระบวนการย่อย (subprocesses) ที่แยกส่วนกันอย่างสมบูรณ์ และล้าง CUDA contexts ทั้งหมดระหว่างสลับไฟล์ เพื่อป้องกันปัญหา VRAM fragmentation และอาการแครชจาก OOM (โดยเฉพาะใน Docker/WSL)
- การให้คะแนนแอ็กชันจากเสียง + วิดีโอ:
  - การจัดอันดับแบบผสมผสานโดยสามารถปรับน้ำหนักได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- จัดอันดับฉากตามคะแนนแอ็กชันรวมแทนที่จะเป็นระยะเวลา
- **การตัดฉากอัจฉริยะ (Smart Scene Cutting)**:
  - เลือกฉากที่สมบูรณ์เป็นหลักหากอยู่ในระยะเวลาที่จำกัด
  - **การเสริมความยาวฉาก (Scene Padding)**: เพิ่มบัฟเฟอร์ 1.5 วินาทีที่ตอนท้ายของฉาก เพื่อเก็บบันทึกภาพแอนิเมชันตอนจบและการเฟดภาพ
  - **การตัดแต่งอัจฉริยะ (Smart Trimming)**: สำหรับฉากยาวๆ จะค้นหาช่วงเวลาที่ "เงียบ" (เสียง/การเคลื่อนไหวน้อย) ในการตัด เพื่อหลีกเลี่ยงการจบแบบกะทันหัน
- การครอบตัดอัจฉริยะ พร้อมตัวเลือกการใส่พื้นหลังเบลอสำหรับฟุตเทจที่ไม่ใช่วิดีโอแนวตั้ง
- ระบบลองทำซ้ำ (Retry logic) ระหว่างการเรนเดอร์เพื่อหลีกเลี่ยงข้อผิดพลาดที่ผิดปกติ
- ตั้งค่าผ่านตัวแปรสภาพแวดล้อมในไฟล์ `.env`

## 📋 ความต้องการของระบบ (Requirements)

- **NVIDIA GPU** ที่รองรับ CUDA
- **NVIDIA Drivers** (แนะนำเวอร์ชันที่เข้ากันได้กับ CUDA 13.0 ขึ้นไป)
- Python 3.12 ขึ้นไป
- FFmpeg (ใช้สำหรับการแยกเสียงและการเข้ารหัส NVENC)
- System libraries: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีคอมพิวเตอร์วิทัศน์)

ไลบรารี Python ที่ต้องใช้ (ดูเพิ่มเติมใน `pyproject.toml`):
- `torch`, `torchaudio` (ที่รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และชุดเครื่องมือ CUDA เรียบร้อยแล้ว จากนั้นจึงติดตั้งแพ็กเกจโดยตรง:

```bash
pip install shorts-maker-gpu
```

### ตั้งค่าแบบแมนนวลจากซอร์สโค้ด (Linux พร้อม CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และชุดเครื่องมือ CUDA เรียบร้อยแล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# ติดตั้งไลบรารีและส่วนที่เกี่ยวข้อง
pip install -e .
```

หากคุณพบปัญหา PyTorch ไม่สามารถค้นหา GPU เจอ ให้อ้างอิงจากคู่มือการติดตั้ง PyTorch ให้ตรงกับเวอร์ชัน CUDA ของคุณ

## 💡 การใช้งาน

1. นำวิดีโอต้นฉบับไปวางไว้ในไดเรกทอรี `gameplay/`
2. รันเครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถเลือกกำหนดไดเรกทอรีต้นทาง (input) ปลายทาง (output) และจำกัดจำนวนฉาก (scene limits) ได้เอง:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างเสร็จแล้วจะถูกบันทึกไว้ในไดเรกทอรี `generated/`

ในระหว่างการประมวลผล ไฟล์ล็อกจะแสดงคะแนนแอ็กชันของฉากรวมแต่ละฉาก และรายชื่อสุดท้ายที่ถูกจัดอันดับตามคะแนนดังกล่าว ฉากที่อยู่ด้านบนสุด (ตามความเข้มข้นของแอ็กชัน) จะถูกเรนเดอร์ก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**ข้อกำหนดเบื้องต้น**: ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) ลงในเครื่องโฮสต์ก่อน

บิลด์และรัน:

*(หมายเหตุ: หากบิลด์เกิดขัดข้องโดยมีข้อความว่า "Segmentation fault" หรือข้อผิดพลาดเกี่ยวกับหน่วยความจำ ให้จำกัดจำนวนคอร์ CPU โดยใช้คำสั่ง `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

```bash
docker build -t shorts-maker .

# รันโดยเปิดสิทธิ์การเข้าถึง GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

โปรดสังเกตแฟล็ก `--gpus all` ซึ่งจำเป็นอย่างมากเพื่อให้แอปพลิเคชันเข้าถึงการเร่งความเร็วทางฮาร์ดแวร์ได้

## ⚙️ การตั้งค่า (Configuration)

คัดลอกไฟล์ `.env.example` เป็น `.env` และปรับค่าตามที่ต้องการ

ตัวแปรที่รองรับ (พร้อมแสดงค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — อัตราส่วนความกว้างของภาพที่ต้องการ (เช่น 9 สำหรับอัตราส่วน 9:16)
- `TARGET_RATIO_H=16` — อัตราส่วนความสูงของภาพที่ต้องการ (เช่น 16 สำหรับอัตราส่วน 9:16)
- `SCENE_LIMIT=4` — จำนวนสูงสุดของฉากยอดนิยมที่จะทำการเรนเดอร์ต่อหนึ่งวิดีโอต้นฉบับ
- `SCENE_THRESHOLD=45.0` — ค่าความแม่นยำ (Threshold) สำหรับการตัดแบ่งฉาก
- `X_CENTER=0.5` — จุดกึ่งกลางการครอบตัดแนวนอน ในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — จุดกึ่งกลางการครอบตัดแนวตั้ง ในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดที่ให้ลองใหม่หากการเรนเดอร์ล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาววิดีโอสั้นขั้นต่ำ (หน่วยเป็นวินาที)
- `MAX_SHORT_LENGTH=179` — ความยาววิดีโอสั้นสูงสุด (หน่วยเป็นวินาที)
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวรวมของฉากสูงสุด (หน่วยเป็นวินาที)
- `SAVE_FFMPEG_LOGS=False` — กำหนดว่าต้องการบันทึกไฟล์ล็อก FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับการบันทึกข้อมูลล็อก (เช่น INFO, DEBUG, WARNING)

## 🛠️ การพัฒนา (Development)

### การตรวจสอบโค้ด (Linting)

โปรเจกต์นี้ใช้ `ruff` เพื่อการตรวจสอบโค้ดที่รวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การรันทดสอบ (Running Tests)

Unit tests จะอยู่ในโฟลเดอร์ `tests/` คุณสามารถรันทดสอบได้ด้วยคำสั่ง:

```bash
pytest -q
```

หมายเหตุ: ชุดทดสอบถูกออกแบบมาให้จำลอง (mock) การทำงานของ GPU กรณีที่ไม่มีฮาร์ดแวร์อยู่จริง เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ปัญหาเบื้องต้น (Troubleshooting)

- **ข้อผิดพลาด "internal compiler error: Segmentation fault" ระหว่างรัน `docker build`**: ปัญหานี้มักจะเกิดขึ้นจากข้อผิดพลาด Out-Of-Memory (OOM) เมื่อ Docker พยายามจะคอมไพล์ไลบรารี C++/CUDA ที่มีขนาดใหญ่ (เช่น VPF) โดยใช้คอร์ CPU ทั้งหมดที่มี เพื่อแก้ไขปัญหานี้ ให้จำกัดจำนวนคอร์ CPU ที่ใช้ในขั้นตอนการบิลด์:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือคุณอาจเพิ่มขีดจำกัด RAM สำหรับ Docker/WSL2 ในการตั้งค่าระบบของคุณแทนก็ได้)*
- **"Torch not installed" / "CUDA not available"**: ให้แน่ใจว่าคุณรัน Docker container ร่วมกับคำสั่ง `--gpus all` หรือได้ติดตั้ง CUDA toolkit ที่ถูกต้องลงบนเครื่องแล้ว
- **ข้อผิดพลาดเกี่ยวกับ NVENC**: หากรัน `h264_nvenc` ล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสแบบซอฟต์แวร์ (`libx264`) ให้คุณตรวจสอบให้แน่ใจว่า GPU ของคุณรองรับ NVENC และอัปเดตไดรเวอร์ให้เป็นเวอร์ชันล่าสุด

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้เผยแพร่ภายใต้สัญญาอนุญาต [MIT License](LICENSE)