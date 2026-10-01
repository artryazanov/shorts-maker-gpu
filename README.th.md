> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (ปรับแต่งสำหรับ GPU)

Shorts Maker จะสร้างคลิปวิดีโอแนวตั้งจากฟุตเทจเกมเพลย์ที่มีความยาว ไลบรารี Python และเครื่องมือ CLI นี้จะตรวจจับฉาก คำนวณโปรไฟล์แอ็กชันของเสียงและวิดีโอ (ความดังของเสียง + การเคลื่อนไหวของภาพ) และนำมารวมกันเพื่อจัดอันดับฉากตามความเข้มข้นโดยรวม จากนั้นระบบจะครอบตัดตามอัตราส่วนภาพที่ต้องการและเรนเดอร์วิดีโอสั้น (Shorts) ที่พร้อมอัปโหลด

**เวอร์ชันนี้ได้รับการปรับแต่งประสิทธิภาพอย่างมากสำหรับ NVIDIA GPU โดยใช้ CUDA**

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

## ✨ คุณสมบัติ

- **การประมวลผลด้วยการเร่งความเร็วด้วย GPU**:
  - **การถอดรหัสและการปรับขนาดด้วยฮาร์ดแวร์**: บูรณาการเข้ากับ NVIDIA Video Processing Framework (VPF) แบบเนทีฟผ่าน `PyNvCodec` สามารถถอดรหัส ปรับขนาด และแปลงปริภูมิสีได้โดยตรงบน NVDEC
  - **การตรวจจับฉาก**: การนำไปใช้งานแบบกำหนดเองโดยใช้ VPF และ OpenCV
  - **การวิเคราะห์เสียง**: ใช้ `torchaudio` บน GPU เพื่อการคำนวณ RMS และ spectral flux อย่างรวดเร็ว
  - **การวิเคราะห์วิดีโอ**: การสตรีมหน่วยความจำ GPU แบบ Zero-copy เพื่อการประเมินการเคลื่อนไหวที่เสถียร (แทนที่การสร้างดัชนีเฟรมที่กินทรัพยากรหนัก)
  - **การประมวลผลภาพ**: ใช้ตัวดำเนินการ PyTorch แบบเนทีฟสำหรับงานหนักๆ เช่น การเบลอพื้นหลัง (separable convolutions)
  - **การเรนเดอร์**: เอนจินแบบกำหนดเองของ PyTorch+NVENC เพื่อการเรนเดอร์ประสิทธิภาพสูง (ถอด MoviePy ออกจากขั้นตอนการเรนเดอร์)
  - **การประมวลผลเป็นชุดที่แข็งแกร่ง**: การประมวลผลวิดีโอจะทำงานในกระบวนการย่อยที่แยกส่วนกันอย่างสมบูรณ์ เพื่อล้าง Context ของ CUDA ระหว่างไฟล์โดยสิ้นเชิง เพื่อป้องกันปัญหาการแตกกระจายของ VRAM และการเกิดข้อผิดพลาด OOM (โดยเฉพาะใน Docker/WSL)
  - **การจัดการ VFR ที่แม่นยำ**: ดึงข้อมูล Presentation Timestamps (PTS) ที่แท้จริงจากแพ็กเก็ตวิดีโอโดยตรงเพื่อป้องกันปัญหาเสียง/วิดีโอไม่ตรงกัน (desync) สามารถจัดการเกมเพลย์แบบ Variable Frame Rate (VFR) ได้อย่างไร้รอยต่อ
- การให้คะแนนแอ็กชันของเสียง + วิดีโอ:
  - จัดอันดับแบบผสมผสานพร้อมน้ำหนักที่ปรับแต่งได้ (ค่าเริ่มต้น: เสียง 0.6, วิดีโอ 0.4)
- ฉากต่างๆ จะถูกจัดอันดับตามคะแนนแอ็กชันแบบผสมแทนที่จะจัดตามระยะเวลา
- **การตัดฉากแบบอัจฉริยะ**:
  - เลือกฉากที่สมบูรณ์เป็นหลักหากความยาวอยู่ในขีดจำกัดเวลา
  - **การเพิ่มบัฟเฟอร์ในฉาก**: เพิ่มบัฟเฟอร์ 1.5 วินาทีต่อท้ายฉากเพื่อให้จับภาพแอนิเมชันและการเฟดตอนจบได้
  - **การตัดแต่งแบบอัจฉริยะ**: สำหรับฉากที่ยาว จะค้นหาช่วงเวลาที่ "เงียบ" (เสียง/การเคลื่อนไหวน้อย) เพื่อทำการตัด หลีกเลี่ยงการจบแบบกระทันหัน
- การครอบตัดอัจฉริยะพร้อมตัวเลือกพื้นหลังเบลอสำหรับฟุตเทจที่ไม่ใช่แนวตั้ง
- ตรรกะการลองใหม่ (Retry) ระหว่างการเรนเดอร์เพื่อหลีกเลี่ยงข้อผิดพลาดที่เกิดขึ้นแบบสุ่ม
- การกำหนดค่าผ่านตัวแปรสภาพแวดล้อม `.env`

## 📋 ความต้องการของระบบ

- **NVIDIA GPU** ที่รองรับ CUDA
- **ไดรเวอร์ NVIDIA** (แนะนำรุ่นที่เข้ากันได้กับ CUDA 13.0 ขึ้นไป)
- Python 3.12+
- FFmpeg (ใช้สำหรับการดึงเสียงและการเข้ารหัสด้วย NVENC)
- ไลบรารีระบบ: `libgl1`, `libglib2.0-0` (มักจำเป็นสำหรับไลบรารีทางด้านวิชัน)

การพึ่งพาไลบรารีของ Python (ดูที่ `pyproject.toml`):
- `torch`, `torchaudio` (ที่รองรับ CUDA)
- `PyNvCodec`, `PytorchNvCodec` (Video Processing Framework)

## 🚀 การติดตั้ง

### ผ่าน PyPI (แนะนำ)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และ CUDA toolkit แล้ว จากนั้นติดตั้งแพ็กเกจโดยตรง:

```bash
pip install shorts-maker-gpu
```

### การติดตั้งด้วยตนเองจาก Source (Linux ที่มี CUDA)

ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไดรเวอร์ NVIDIA และ CUDA toolkit แล้ว

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

หากคุณพบปัญหาที่ PyTorch หา GPU ไม่พบ โปรดอ้างอิงคู่มือการติดตั้งให้ตรงกับเวอร์ชัน CUDA เฉพาะของคุณ

## 💡 การใช้งาน

1. วางวิดีโอต้นฉบับไว้ในไดเรกทอรี `gameplay/`
2. เรียกใช้เครื่องมือ CLI:

```bash
shorts-maker process
```

คุณสามารถเลือกปรับแต่งไดเรกทอรีอินพุตและเอาต์พุต รวมทั้งขีดจำกัดจำนวนฉากได้:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. คลิปที่สร้างขึ้นจะถูกบันทึกไว้ในไดเรกทอรี `generated/`

ระหว่างการประมวลผล บันทึก (log) จะแสดงคะแนนแอ็กชันสำหรับแต่ละฉากที่นำมารวมกัน และแสดงรายการสุดท้ายที่จัดเรียงตามคะแนนนั้น ฉากที่ดีที่สุด (จัดตามความเข้มข้นของแอ็กชัน) จะถูกเรนเดอร์ก่อนโดยใช้ NVENC

## 🐳 Docker (แนะนำ)

วิธีที่ง่ายที่สุดในการรันแอปพลิเคชันนี้คือการใช้ Docker ร่วมกับ NVIDIA Container Toolkit

**ข้อกำหนดเบื้องต้น**: ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) ไว้บนโฮสต์

วิธีสร้างและรัน:

*(หมายเหตุ: หากการสร้าง (build) แครชและแสดงข้อผิดพลาด "Segmentation fault" หรือมีข้อผิดพลาดเกี่ยวกับหน่วยความจำ ให้จำกัดแกน CPU โดยใช้ `docker build --cpuset-cpus="0,1" -t shorts-maker .` แทน)*

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

โปรดสังเกตการตั้งค่าแฟล็ก `--gpus all` ซึ่งมีความจำเป็นสำหรับแอปพลิเคชันเพื่อเข้าถึงการเร่งความเร็วด้วยฮาร์ดแวร์

## ⚙️ การตั้งค่า

คัดลอกไฟล์ `.env.example` เป็น `.env` และปรับเปลี่ยนค่าตามต้องการ

ตัวแปรที่รองรับ (แสดงค่าเริ่มต้น):
- `TARGET_RATIO_W=9` — สัดส่วนความกว้างของอัตราส่วนภาพเป้าหมาย (เช่น 9 สำหรับ 9:16)
- `TARGET_RATIO_H=16` — สัดส่วนความสูงของอัตราส่วนภาพเป้าหมาย (เช่น 16 สำหรับ 9:16)
- `SCENE_LIMIT=4` — จำนวนฉากที่ดีที่สุดสูงสุดที่จะนำมาเรนเดอร์ต่อหนึ่งวิดีโอต้นฉบับ
- `SCENE_THRESHOLD=45.0` — เกณฑ์ขั้นต่ำ (Threshold) สำหรับการตัดในระบบตรวจจับฉาก
- `X_CENTER=0.5` — จุดกึ่งกลางของการครอบตัดแนวนอนในช่วง [0.0, 1.0]
- `Y_CENTER=0.5` — จุดกึ่งกลางของการครอบตัดแนวตั้งในช่วง [0.0, 1.0]
- `MAX_ERROR_DEPTH=3` — จำนวนครั้งสูงสุดที่จะลองใหม่ (retry depth) ในกรณีที่การเรนเดอร์ล้มเหลว
- `MIN_SHORT_LENGTH=15` — ความยาววิดีโอสั้นขั้นต่ำ (หน่วยเป็นวินาที)
- `MAX_SHORT_LENGTH=179` — ความยาววิดีโอสั้นสูงสุด (หน่วยเป็นวินาที)
- `MAX_COMBINED_SCENE_LENGTH=300` — ความยาวรวมสูงสุดของฉาก (หน่วยเป็นวินาที)
- `SKIP_FIRST_SECONDS=0.0` — จำนวนวินาทีที่ต้องการข้ามจากช่วงเริ่มต้นของวิดีโอ (มีประโยชน์สำหรับการข้ามหน้าจออินโทร)
- `SAVE_FFMPEG_LOGS=False` — กำหนดว่าจะบันทึก log ของ FFmpeg ระหว่างการเรนเดอร์หรือไม่
- `LOG_LEVEL=WARNING` — ระดับการบันทึกข้อมูล (เช่น INFO, DEBUG, WARNING)

## 🛠️ การพัฒนา

### การทำ Linting

โปรเจกต์นี้ใช้ `ruff` เพื่อการทำ linting อย่างรวดเร็ว

```bash
pip install ruff
ruff check .
```

## 🧪 การรันการทดสอบ

Unit tests อยู่ในโฟลเดอร์ `tests/` รันด้วยคำสั่ง:

```bash
pytest -q
```

หมายเหตุ: การทดสอบออกแบบมาให้จำลอง (mock) สถานะของ GPU ในกรณีที่ไม่มีให้ใช้งาน เพื่อให้สามารถรันในสภาพแวดล้อม CI มาตรฐานได้

## 🚑 การแก้ไขปัญหา

- **เกิดข้อผิดพลาด "internal compiler error: Segmentation fault" ระหว่าง `docker build`**: มักเกิดขึ้นเนื่องจากปัญหาหน่วยความจำไม่เพียงพอ (Out-Of-Memory: OOM) เมื่อ Docker พยายามคอมไพล์ไลบรารี C++/CUDA ขนาดใหญ่ (เช่น VPF) โดยใช้แกน CPU ทั้งหมดที่มี วิธีแก้ไขคือการจำกัดจำนวนแกน CPU ที่ใช้ระหว่างขั้นตอนการ build:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(หรือคุณอาจเลือกที่จะเพิ่มขีดจำกัดของ RAM สำหรับ Docker/WSL2 ในการตั้งค่าระบบของคุณก็ได้)*
- **"WSL integration with distro unexpectedly stopped" / ข้อผิดพลาด OOM ระหว่างที่รัน `docker run`**: การประมวลผลวิดีโอความละเอียดสูงจะกินทรัพยากร RAM/VRAM จำนวนมาก ทำให้เครื่องเสมือนของ WSL2 แครชจากข้อผิดพลาดหน่วยความจำไม่เพียงพอ (OOM) วิธีแก้ไขคือการจำกัดจำนวนแกน CPU ที่คอนเทนเนอร์สามารถใช้งานได้ขณะประมวลผลโดยการเพิ่มแฟล็ก `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: ให้แน่ใจว่าคุณได้รันในคอนเทนเนอร์ Docker พร้อมด้วย `--gpus all` หรือได้ติดตั้ง CUDA toolkit รุ่นที่ถูกต้องลงในเครื่องแล้ว
- **ข้อผิดพลาดเกี่ยวกับ NVENC**: หาก `h264_nvenc` ล้มเหลว สคริปต์จะพยายามกลับไปใช้การเข้ารหัสด้วยซอฟต์แวร์ (`libx264`) แทน โปรดตรวจสอบว่า GPU ของคุณรองรับ NVENC และไดรเวอร์เป็นเวอร์ชันล่าสุดหรือไม่

## 📄 สิทธิการใช้งาน (License)

โปรเจกต์นี้เผยแพร่ภายใต้ [MIT License](LICENSE)