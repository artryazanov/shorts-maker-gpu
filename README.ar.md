> 🌐 **اللغات:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 صانع الفيديوهات القصيرة Shorts Maker (مُحسّن لوحدات معالجة الرسومات GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو رأسية (Shorts) من لقطات اللعب (gameplay) الطويلة. تكتشف مكتبة Python وأداة سطر الأوامر (CLI) هذه المشاهد، وتحسب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، وتدمجها لتصنيف المشاهد بناءً على الشدة الكلية. بعد ذلك، تقوم باقتطاع الفيديو ليناسب نسبة العرض إلى الارتفاع المطلوبة وتُخرج فيديوهات قصيرة جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لوحدات معالجة الرسومات NVIDIA باستخدام CUDA.**

للحصول على النسخة الأصلية التي تعتمد على المعالج المركزي (CPU) فقط، يرجى زيارة [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

- **معالجة مُسرَّعة بواسطة وحدة معالجة الرسومات (GPU)**:
  - **فك التشفير وتغيير الحجم العتادي (Hardware Decoding & Resizing)**: تكامل أصلي مع إطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير، وتغيير الحجم، وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مُخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات لحساب سريع لجذر متوسط المربع (RMS) والتدفق الطيفي (spectral flux).
  - **تحليل الفيديو**: تدفق مباشر عبر ذاكرة وحدة معالجة الرسومات بدون نسخ (Zero-copy) لتقدير الحركة بشكل مستقر (يستبدل فهارس الإطارات الثقيلة).
  - **معالجة الصور**: استخدام مُشغّلات PyTorch الأصلية للعمليات الثقيلة مثل تعتيم الخلفيات (التلافيف القابلة للفصل - separable convolutions).
  - **التقديم (Rendering)**: محرك PyTorch+NVENC مُخصص لتقديم عالي الأداء (تمت إزالة MoviePy من مسار التقديم).
  - **معالجة دفعية قوية (Robust Batch Processing)**: تعمل معالجة الفيديو في عمليات فرعية معزولة تمامًا، مما يؤدي إلى مسح سياقات CUDA بالكامل بين الملفات لمنع تجزئة ذاكرة الوصول العشوائي للفيديو (VRAM) وأعطال نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
- تقييم الحركة الصوتية + المرئية:
  - تصنيف مُدمج بأوزان قابلة للتعديل (الافتراضي: الصوت 0.6، الفيديو 0.4).
- تصنيف المشاهد بناءً على درجة الحركة المُدمجة بدلاً من المدة.
- **قص ذكي للمشاهد**:
  - يُفضل تحديد المشاهد الكاملة إذا كانت تتناسب مع الحد الزمني.
  - **حشو المشهد (Scene Padding)**: إضافة مخزن مؤقت (buffer) مدته 1.5 ثانية إلى نهاية المشاهد لالتقاط الحركات الختامية (exit animations) والتلاشي.
  - **اقتطاع ذكي (Smart Trimming)**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (حيث ينخفض مستوى الصوت/الحركة) للقص، لتجنب النهايات المفاجئة.
- اقتصاص ذكي مع خلفية ضبابية اختيارية للقطات غير الرأسية.
- منطق إعادة المحاولة (Retry logic) أثناء التقديم (rendering) لتجنب الأخطاء العابرة.
- التهيئة والإعداد عبر متغيرات البيئة `.env`.

## 📋 المتطلبات

- **وحدة معالجة الرسومات NVIDIA** مع دعم CUDA.
- **تعريفات NVIDIA** (يوصى بالإصدارات المتوافقة مع CUDA 13.0+).
- بايثون (Python) 3.12+
- أداة FFmpeg (تُستخدم لاستخراج الصوت وترميز NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالبًا ما تكون مطلوبة لمكتبات الرؤية الحاسوبية).

تبعيات بايثون (راجع `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (مستحسن)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA. ثم قم بتثبيت الحزمة مباشرة:

```bash
pip install shorts-maker-gpu
```

### الإعداد اليدوي من المصدر (نظام Linux مع CUDA)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# تثبيت المكتبة وتبعياتها
pip install -e .
```

إذا واجهت مشاكل تتعلق بعدم عثور PyTorch على وحدة معالجة الرسومات، راجع دليل التثبيت الخاص به لإصدار CUDA المحدد لديك.

## 💡 الاستخدام

1. ضع مقاطع الفيديو المصدر داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك بشكل اختياري تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يُظهر السجل درجة الحركة (action score) لكل مشهد مُدمج والقائمة النهائية مُرتبة بناءً على تلك الدرجة. يتم تقديم (render) المشاهد الأعلى (من حيث شدة الحركة) أولاً باستخدام NVENC.

## 🐳 دوكر Docker (مستحسن)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع مجموعة أدوات الحاويات من NVIDIA (NVIDIA Container Toolkit).

**شرط أساسي**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على النظام المضيف.

البناء والتشغيل:

*(ملاحظة: إذا تعطل البناء مع ظهور خطأ "Segmentation fault" أو خطأ في الذاكرة، فقم بتقييد نوى وحدة المعالجة المركزية (CPU) باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

```bash
docker build -t shorts-maker .

# التشغيل مع الوصول إلى وحدة معالجة الرسومات
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

لاحظ وجود العلامة `--gpus all`، وهي ضرورية حتى يتمكن التطبيق من الوصول إلى تسريع الأجهزة (hardware acceleration).

## ⚙️ التهيئة (Configuration)

انسخ `.env.example` إلى `.env` واضبط القيم حسب الحاجة.

المتغيرات المدعومة (تظهر القيم الافتراضية):
- `TARGET_RATIO_W=9` — جزء العرض من نسبة العرض إلى الارتفاع المستهدفة (على سبيل المثال، 9 لنسبة 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع من نسبة العرض إلى الارتفاع المستهدفة (على سبيل المثال، 16 لنسبة 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد المشاهد الأعلى التي سيتم تقديمها لكل فيديو مصدر.
- `SCENE_THRESHOLD=45.0` — العتبة (Threshold) المستخدمة لاكتشاف تقطيعات المشاهد.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في نطاق [0.0, 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في نطاق [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لمرات إعادة المحاولة إذا فشل التقديم.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول الفيديو القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول الفيديو القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المُدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — ما إذا كان سيتم حفظ سجلات FFmpeg أثناء عملية التقديم.
- `LOG_LEVEL=WARNING` — مستوى التسجيل (مثلاً: INFO, DEBUG, WARNING).

## 🛠️ التطوير

### فحص جودة الكود (Linting)

يستخدم هذا المشروع `ruff` لفحص جودة الكود بسرعة.

```bash
pip install ruff
ruff check .
```

## 🧪 تشغيل الاختبارات

توجد اختبارات الوحدة (Unit tests) في مجلد `tests/`. قم بتشغيلها باستخدام:

```bash
pytest -q
```

ملاحظة: تم تصميم الاختبارات لمحاكاة (mock) توافر وحدة معالجة الرسومات في حال غيابها، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **ظهور خطأ "internal compiler error: Segmentation fault" أثناء تشغيل `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع (compile) مكتبات C++/CUDA ثقيلة (مثل VPF) باستخدام جميع نوى المعالج المتاحة. لإصلاح ذلك، قم بتقييد عدد نوى المعالج المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(كبديل، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) المخصصة لـ Docker/WSL2 في إعدادات نظامك).*
- **خطأ "Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker مع استخدام العلم `--gpus all`، أو تأكد من تثبيت مجموعة أدوات CUDA الصحيحة محليًا.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، سيحاول السكربت التراجع إلى استخدام الترميز البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات الخاصة بك تدعم NVENC وما إذا كانت التعريفات مُحدثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).