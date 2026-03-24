> 🌐 **Languages:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 صانع المقاطع القصيرة (Shorts Maker) (محسّن لوحدات معالجة الرسومات GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو عمودية قصيرة من لقطات أسلوب اللعب (gameplay) الطويلة. تكتشف مكتبة Python وأداة سطر الأوامر (CLI) هذه المشاهد، وتحسب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، وتجمعها لترتيب المشاهد حسب الشدة الإجمالية. بعد ذلك، تقوم باقتصاص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وتُصدِّر مقاطع قصيرة جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لتناسب وحدات معالجة الرسومات (GPUs) من NVIDIA باستخدام تقنية CUDA.**

للحصول على النسخة الأصلية التي تعتمد على وحدة المعالجة المركزية (CPU) فقط، يرجى زيارة [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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
  - **فك التشفير وتغيير الحجم العتادي**: دمج أصلي مع إطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير، وتغيير الحجم، وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات لحساب سريع لجذر متوسط المربع (RMS) والتدفق الطيفي.
  - **تحليل الفيديو**: بث بذاكرة وصول عشوائي للفيديو (GPU memory streaming) بدون نسخ لتقدير حركة مستقر (يستبدل مؤشرات الإطارات الثقيلة).
  - **معالجة الصور**: استخدام عمليات PyTorch الأصلية للعمليات الثقيلة مثل تعتيم الخلفيات (الالتواءات القابلة للفصل).
  - **التصيير (Rendering)**: محرك PyTorch+NVENC مخصص لتصيير عالي الأداء (تم إزالة MoviePy من مسار التصيير).
  - **معالجة الدُفعات القوية (Batch Processing)**: تعمل معالجة الفيديو في عمليات فرعية معزولة تمامًا، وتمسح سياقات CUDA بالكامل بين الملفات لمنع تجزئة ذاكرة الفيديو (VRAM) وأعطال نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
- تسجيل أداء الحركة الصوتية + المرئية:
  - ترتيب مُدمج بأوزان قابلة للضبط (القيم الافتراضية: الصوت 0.6، الفيديو 0.4).
- ترتيب المشاهد حسب درجة الحركة المدمجة بدلاً من المدة.
- **قطع ذكي للمشاهد**:
  - يُفضّل اختيار المشاهد الكاملة إذا كانت تتناسب مع الحد الزمني.
  - **حشو المشهد (Padding)**: يضيف مساحة مؤقتة مدتها 1.5 ثانية إلى نهاية المشاهد لالتقاط حركات الخروج والتلاشي.
  - **تشذيب ذكي (Trimming)**: بالنسبة للمشاهد الطويلة، يبحث عن لحظات "هادئة" (انخفاض الصوت/الحركة) لقطعها، لتجنب النهايات المفاجئة.
- اقتصاص ذكي مع خلفية ضبابية اختيارية للقطات غير العمودية.
- منطق إعادة المحاولة أثناء التصيير لتجنب حالات الفشل العابرة.
- الإعدادات عبر متغيرات البيئة `.env`.

## 📋 المتطلبات

- **وحدة معالجة رسومات (GPU) من NVIDIA** تدعم تقنية CUDA.
- **تعريفات NVIDIA** (يُوصى بتلك المتوافقة مع CUDA 13.0 وما فوق).
- إصدار Python 3.12 أو أحدث
- FFmpeg (يُستخدم لاستخراج الصوت والترميز عبر NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالبًا ما تكون مطلوبة لمكتبات الرؤية الحاسوبية).

تبعيات Python (انظر ملف `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (مستحسن)

تأكد من تثبيت تعريفات NVIDIA وحزمة أدوات CUDA. ثم قم بتثبيت الحزمة مباشرة:

```bash
pip install shorts-maker-gpu
```

### الإعداد اليدوي من المصدر (نظام Linux مع CUDA)

تأكد من تثبيت تعريفات NVIDIA وحزمة أدوات CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# Install the library and its dependencies
pip install -e .
```

إذا واجهت مشاكل تتعلق بعدم عثور PyTorch على وحدة معالجة الرسومات (GPU)، فارجع إلى دليل التثبيت الخاص به لإصدار CUDA الذي تستخدمه.

## 💡 الاستخدام

1. ضع مقاطع الفيديو المصدرية داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك اختياريًا تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يُظهر السجل درجة حركة لكل مشهد مُدمج والقائمة النهائية مرتبة بناءً على تلك الدرجة. يتم تصيير أفضل المشاهد (من حيث شدة الحركة) أولاً باستخدام NVENC.

## 🐳 استخدام Docker (مستحسن)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع NVIDIA Container Toolkit.

**المتطلبات الأساسية**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا تعطلت عملية البناء برسالة "Segmentation fault" أو بخطأ في الذاكرة، فقم بتحديد أنوية وحدة المعالجة المركزية باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

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

لاحظ وجود العلامة `--gpus all`، وهي ضرورية لكي يتمكن التطبيق من الوصول إلى التسريع العتادي.

## ⚙️ الإعدادات (Configuration)

انسخ ملف `.env.example` إلى `.env` واضبط القيم حسب الحاجة.

المتغيرات المدعومة (القيم الافتراضية معروضة):
- `TARGET_RATIO_W=9` — الجزء الخاص بالعرض لنسبة العرض إلى الارتفاع المستهدفة (مثلاً 9 لنسبة 9:16).
- `TARGET_RATIO_H=16` — الجزء الخاص بالارتفاع لنسبة العرض إلى الارتفاع المستهدفة (مثلاً 16 لنسبة 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد أفضل المشاهد التي يتم تصييرها لكل فيديو مصدري.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في النطاق [0.0، 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في النطاق [0.0، 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لمرات إعادة المحاولة إذا فشل التصيير.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول المقطع القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول المقطع القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المُدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — لتحديد ما إذا كان سيتم حفظ سجلات FFmpeg أثناء التصيير.
- `LOG_LEVEL=WARNING` — مستوى التسجيل (مثل INFO، DEBUG، WARNING).

## 🛠️ التطوير

### فحص جودة الكود (Linting)

يستخدم هذا المشروع أداة `ruff` لفحص جودة الكود بشكل سريع.

```bash
pip install ruff
ruff check .
```

## 🧪 تشغيل الاختبارات

توجد اختبارات الوحدة (Unit tests) في مجلد `tests/`. قم بتشغيلها باستخدام:

```bash
pytest -q
```

ملاحظة: تم تصميم الاختبارات لمحاكاة (mock) توفر وحدة معالجة الرسومات في حالة عدم وجودها، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **خطأ المترجم الداخلي: "internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ في نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA ثقيلة (مثل VPF) باستخدام جميع أنوية وحدة المعالجة المركزية المتاحة. لإصلاح ذلك، قم بتقليل عدد أنوية وحدة المعالجة المركزية المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(بدلاً من ذلك، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) المخصص لـ Docker/WSL2 في إعدادات النظام الخاص بك).*
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker مع العلامة `--gpus all` أو أنك قمت بتثبيت حزمة أدوات CUDA الصحيحة محليًا.
- **خطأ في NVENC**: إذا فشل `h264_nvenc`، سيحاول السكربت العودة إلى الترميز البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات لديك تدعم NVENC وما إذا كانت التعريفات محدثة.

## 📄 الترخيص

تم إصدار هذا المشروع تحت [ترخيص MIT](LICENSE).