> 🌐 **اللغات:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (مُحسَّن لوحدات معالجة الرسومات GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو رأسية قصيرة من لقطات ألعاب الفيديو الطويلة. تكتشف هذه المكتبة المكتوبة بلغة بايثون وأداة سطر الأوامر (CLI) المشاهد، وتحسب ملفات تعريف الحركة للصوت والفيديو (كثافة الصوت + الحركة البصرية)، وتدمجها لترتيب المشاهد بناءً على الكثافة الإجمالية. ثم تقوم باقتصاص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وتصدير مقاطع قصيرة (Shorts) جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لوحدات معالجة الرسومات (GPUs) من NVIDIA باستخدام CUDA.**

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

- **معالجة مسرعة بوحدة معالجة الرسومات (GPU)**:
  - **فك التشفير وتغيير الحجم العتادي**: دمج أصلي لإطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يفك التشفير ويغير الحجم ويحول مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات لحساب سريع لجذر متوسط المربع (RMS) والتدفق الطيفي.
  - **تحليل الفيديو**: تدفق ذاكرة وحدة معالجة الرسومات بدون نسخ (Zero-copy) لتقدير الحركة بشكل مستقر (يستبدل فهارس الإطارات الثقيلة).
  - **معالجة الصور**: استخدام عمليات PyTorch الأصلية للعمليات الثقيلة مثل تمويه الخلفيات (الالتواءات القابلة للفصل).
  - **التصيير (Rendering)**: محرك PyTorch+NVENC مخصص للتصيير عالي الأداء (تم إزالة MoviePy من مسار التصيير).
  - **معالجة مجمّعة قوية**: تعمل معالجة الفيديو في عمليات فرعية معزولة بالكامل، مما يؤدي إلى مسح سياقات CUDA تمامًا بين الملفات لمنع تجزئة ذاكرة الوصول العشوائي للفيديو (VRAM) وأعطال نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
  - **معالجة دقيقة لمعدل الإطارات المتغير (VFR)**: يستخرج الطوابع الزمنية الحقيقية للعرض (PTS) مباشرة من حزم الفيديو لمنع عدم تزامن الصوت/الفيديو، مما يعالج لقطات الألعاب ذات معدل الإطارات المتغير بسلاسة.
- تقييم حركة الصوت والفيديو:
  - تصنيف مدمج بأوزان قابلة للتعديل (الافتراضي: الصوت 0.6، الفيديو 0.4).
- ترتيب المشاهد بناءً على درجة الحركة المدمجة بدلاً من المدة.
- **قص المشاهد الذكي**:
  - يفضل تحديد المشاهد الكاملة إذا كانت تتناسب مع الحد الزمني.
  - **توسيد المشهد**: يضيف مخزنًا مؤقتًا (buffer) مدته 1.5 ثانية في نهاية المشاهد لالتقاط حركات الخروج والتلاشي.
  - **تشذيب ذكي**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (صوت/حركة منخفضة) للقص عندها، متجنبًا النهايات المفاجئة.
- اقتصاص ذكي مع خلفية مموهة اختيارية للقطات غير الرأسية.
- منطق إعادة المحاولة أثناء التصيير لتجنب حالات الفشل العرضية.
- الإعداد عبر متغيرات البيئة `.env`.

## 📋 المتطلبات

- **وحدة معالجة رسومات (GPU) من NVIDIA** مع دعم CUDA.
- **برامج تشغيل NVIDIA** (يوصى ببرامج متوافقة مع CUDA 13.0+).
- بايثون 3.12+
- FFmpeg (يُستخدم لاستخراج الصوت وتشفير NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالبًا ما تكون مطلوبة لمكتبات الرؤية).

اعتماديات بايثون (راجع `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (موصى به)

تأكد من تثبيت برامج تشغيل NVIDIA وحزمة أدوات CUDA. ثم قم بتثبيت الحزمة مباشرة:

```bash
pip install shorts-maker-gpu
```

### الإعداد اليدوي من المصدر (نظام Linux مع CUDA)

تأكد من تثبيت برامج تشغيل NVIDIA وحزمة أدوات CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# تثبيت المكتبة واعتمادياتها
pip install -e .
```

إذا واجهت مشاكل في عدم عثور PyTorch على وحدة معالجة الرسومات، فراجع دليل التثبيت الخاص بها لإصدار CUDA المحدد لديك.

## 💡 الاستخدام

1. ضع مقاطع الفيديو المصدر داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك اختياريًا تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يُظهر السجل درجة الحركة لكل مشهد مدمج والقائمة النهائية مرتبة حسب تلك الدرجة. يتم تصيير المشاهد الأفضل (حسب كثافة الحركة) أولاً باستخدام NVENC.

## 🐳 دوكر Docker (موصى به)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع NVIDIA Container Toolkit.

**المتطلبات الأساسية**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا تعطل البناء برسالة "Segmentation fault" أو خطأ في الذاكرة، فقم بتقييد أنوية وحدة المعالجة المركزية باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

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

لاحظ وجود العلامة `--gpus all`، وهي ضرورية لكي يصل التطبيق إلى تسريع الأجهزة.

## ⚙️ الإعدادات

انسخ `.env.example` إلى `.env` وقم بتعديل القيم حسب الحاجة.

المتغيرات المدعومة (القيم الافتراضية موضحة):
- `TARGET_RATIO_W=9` — جزء العرض من نسبة العرض إلى الارتفاع المستهدفة (مثال: 9 لـ 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع من نسبة العرض إلى الارتفاع المستهدفة (مثال: 16 لـ 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد أفضل المشاهد التي يتم تصييرها لكل فيديو مصدر.
- `SCENE_THRESHOLD=45.0` — الحد الأدنى لقص واكتشاف المشاهد.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في النطاق [0.0، 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في النطاق [0.0، 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لمرات إعادة المحاولة في حال فشل التصيير.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول المقطع القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول المقطع القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — تحديد ما إذا كان سيتم حفظ سجلات FFmpeg أثناء التصيير.
- `LOG_LEVEL=WARNING` — مستوى التسجيل (مثل INFO، DEBUG، WARNING).

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

ملاحظة: صُممت الاختبارات لمحاكاة توفر وحدة معالجة الرسومات إذا كانت مفقودة، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **"internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA الثقيلة (مثل VPF) باستخدام جميع أنوية المعالج المتاحة. لإصلاح ذلك، قم بتقييد عدد الأنوية المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(كبديل، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) المخصصة لـ Docker/WSL2 في إعدادات نظامك).*
- **"WSL integration with distro unexpectedly stopped" / OOM أثناء `docker run`**: يمكن أن تستهلك معالجة مقاطع الفيديو عالية الدقة قدرًا كبيرًا من ذاكرة RAM/VRAM، مما يؤدي إلى تعطل الجهاز الظاهري WSL2 بسبب خطأ نفاد الذاكرة (OOM). لإصلاح ذلك، قم بتقييد عدد أنوية المعالج التي يمكن للحاوية استخدامها أثناء التنفيذ عن طريق إضافة علامة `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تعمل داخل حاوية Docker مع علامة `--gpus all` أو أن لديك حزمة أدوات CUDA الصحيحة مثبتة محليًا.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، سيحاول السكريبت التراجع إلى التشفير البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات لديك تدعم NVENC وما إذا كانت برامج التشغيل محدثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).