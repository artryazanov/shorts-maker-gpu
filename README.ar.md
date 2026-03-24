> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (مُحسّن لمعالجات الرسومات GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو عمودية من لقطات ألعاب الفيديو الطويلة. تقوم هذه المكتبة المكتوبة بلغة Python وأداة سطر الأوامر (CLI) باكتشاف المشاهد، وحساب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، ودمجها لترتيب المشاهد حسب الكثافة الإجمالية. بعد ذلك، تقوم باقتصاص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وتصيير مقاطع قصيرة (shorts) جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لمعالجات الرسومات (GPUs) من NVIDIA باستخدام CUDA.**

للحصول على النسخة الأصلية المخصصة لمعالجات المركزية (CPU) فقط، يرجى زيارة [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

### [اقرأ الوثائق الكاملة 📚](https://artryazanov.github.io/shorts-maker-gpu/)

## ✨ الميزات

- **معالجة مسرّعة بواسطة وحدة معالجة الرسومات (GPU)**:
  - **فك التشفير وتغيير الحجم باستخدام العتاد**: تكامل أصلي مع إطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير، وتغيير الحجم، وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات (GPU) لحساب الجذر التربيعي المتوسط (RMS) والتدفق الطيفي بسرعة.
  - **تحليل الفيديو**: تدفق ذاكرة GPU بدون نسخ (Zero-copy) لتقدير الحركة بشكل مستقر (يستبدل فهارس الإطارات الثقيلة).
  - **معالجة الصور**: عوامل تشغيل PyTorch الأصلية تُستخدم للعمليات الثقيلة مثل تمويه الخلفيات (الالتواءات القابلة للفصل).
  - **التصيير (Rendering)**: محرك مخصص يعتمد على PyTorch+NVENC لتصيير عالي الأداء (تمت إزالة MoviePy من مسار التصيير).
  - **معالجة مجمعة قوية**: تعمل معالجة الفيديو في عمليات فرعية معزولة بالكامل، مما يمسح سياقات CUDA تماماً بين الملفات لمنع تجزئة ذاكرة VRAM وانهيارات نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
- تسجيل الحركة الصوتية والمرئية:
  - ترتيب مدمج مع أوزان قابلة للتعديل (الافتراضي: الصوت 0.6، الفيديو 0.4).
- ترتيب المشاهد حسب درجة الحركة المدمجة بدلاً من المدة الزمنية.
- **قص ذكي للمشاهد**:
  - يفضل اختيار مشاهد كاملة إذا كانت تتناسب مع الحد الزمني.
  - **إضافة حشو للمشاهد (Scene Padding)**: يضيف مساحة تخزين مؤقتة مدتها 1.5 ثانية إلى نهاية المشاهد لالتقاط حركات الخروج والتلاشي.
  - **اقتطاع ذكي**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (انخفاض الصوت/الحركة) للقص عندها، متجنباً النهايات المفاجئة.
- اقتصاص ذكي مع خلفية مموهة اختيارية للقطات غير العمودية.
- منطق إعادة المحاولة أثناء التصيير لتجنب حالات الفشل العابرة.
- التكوين والضبط عبر متغيرات البيئة في ملف `.env`.

## 📋 المتطلبات

- **وحدة معالجة رسومات (GPU) من NVIDIA** مع دعم CUDA.
- **تعريفات NVIDIA** (يوصى بالإصدارات المتوافقة مع CUDA 13.0 فما فوق).
- Python 3.12+
- FFmpeg (يُستخدم لاستخراج الصوت وتشفير NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالباً ما تكون مطلوبة لمكتبات الرؤية الحاسوبية).

اعتماديات Python (انظر `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (مستحسن)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA. ثم قم بتثبيت الحزمة مباشرة:

```bash
pip install shorts-maker-gpu
```

### الإعداد اليدوي من المصدر (Linux مع CUDA)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# تثبيت المكتبة واعتمادياتها
pip install -e .
```

إذا واجهت مشاكل في عدم عثور PyTorch على وحدة معالجة الرسومات (GPU)، فارجع إلى دليل التثبيت الخاص به لإصدار CUDA المحدد لديك.

## 💡 الاستخدام

1. ضع مقاطع الفيديو المصدر داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك اختيارياً تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يوضح السجل درجة الحركة لكل مشهد مدمج والقائمة النهائية مرتبة حسب تلك الدرجة. يتم تصيير أفضل المشاهد (من حيث كثافة الحركة) أولاً باستخدام NVENC.

## 🐳 Docker (مستحسن)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع أداة NVIDIA Container Toolkit.

**المتطلبات الأساسية**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا توقف البناء مع خطأ "Segmentation fault" أو خطأ في الذاكرة، فقم بتقييد عدد أنوية وحدة المعالجة المركزية (CPU) باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

```bash
docker build -t shorts-maker .

# التشغيل مع الوصول إلى GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

لاحظ العلامة `--gpus all`، وهي ضرورية ليتمكن التطبيق من الوصول إلى تسريع العتاد.

## ⚙️ التكوين (Configuration)

انسخ `.env.example` إلى `.env` واضبط القيم حسب الحاجة.

المتغيرات المدعومة (القيم الافتراضية معروضة):
- `TARGET_RATIO_W=9` — جزء العرض من نسبة العرض إلى الارتفاع المستهدفة (مثلاً 9 لنسبة 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع من نسبة العرض إلى الارتفاع المستهدفة (مثلاً 16 لنسبة 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد أفضل المشاهد التي يتم تصييرها لكل فيديو مصدر.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في النطاق [0.0, 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في النطاق [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لعمق إعادة المحاولة في حالة فشل التصيير.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول المقطع القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول المقطع القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — ما إذا كان سيتم حفظ سجلات FFmpeg أثناء التصيير.

## 🛠️ التطوير

### التحقق من الكود (Linting)

يستخدم هذا المشروع `ruff` للتحقق السريع من الكود.

```bash
pip install ruff
ruff check .
```

## 🧪 تشغيل الاختبارات

توجد اختبارات الوحدة (Unit tests) في مجلد `tests/`. قم بتشغيلها بواسطة:

```bash
pytest -q
```

ملاحظة: تم تصميم الاختبارات لمحاكاة توافر وحدة معالجة الرسومات (GPU) في حال عدم وجودها، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **"internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA ثقيلة (مثل VPF) باستخدام جميع أنوية وحدة المعالجة المركزية (CPU) المتاحة. لإصلاح ذلك، قم بتقييد عدد أنوية CPU المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(كبديل، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) لـ Docker/WSL2 في إعدادات نظامك).*
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker مع علامة `--gpus all` أو أنك قمت بتثبيت مجموعة أدوات CUDA الصحيحة محلياً.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، يحاول السكريبت التراجع إلى التشفير البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات لديك تدعم NVENC وما إذا كانت التعريفات مُحدّثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).