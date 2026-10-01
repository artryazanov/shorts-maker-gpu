> 🌐 **اللغات:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 Shorts Maker (مُحسَّن لـ GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو رأسية قصيرة من لقطات اللعب الطويلة. تقوم مكتبة Python وأداة واجهة سطر الأوامر (CLI) هذه باكتشاف المشاهد، وحساب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، ودمجها لترتيب المشاهد بناءً على الكثافة الإجمالية. بعد ذلك، تقوم باقتصاص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وتصيير (render) فيديوهات قصيرة جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لوحدات معالجة الرسومات NVIDIA باستخدام CUDA.**

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
  - **فك التشفير وتغيير الحجم عبر العتاد**: تكامل أصلي مع إطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير، وتغيير الحجم، وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات (GPU) لحسابات جذر متوسط المربع (RMS) والتدفق الطيفي السريعة.
  - **تحليل الفيديو**: بث بذاكرة GPU بنسخ صفري (Zero-copy) لتقدير الحركة بشكل مستقر (يستبدل فهارس الإطارات الثقيلة).
  - **معالجة الصور**: يتم استخدام عمليات PyTorch الأصلية للعمليات الثقيلة مثل تعتيم الخلفيات (التلافيف القابلة للفصل).
  - **التصيير (Rendering)**: محرك PyTorch+NVENC مخصص لتصيير عالي الأداء (تمت إزالة MoviePy من مسار التصيير).
  - **معالجة دفعية قوية**: يتم تشغيل معالجة الفيديو في عمليات فرعية معزولة بالكامل، مما يؤدي إلى مسح سياقات CUDA تمامًا بين الملفات لمنع تجزئة ذاكرة VRAM وانهيارات نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
  - **معالجة دقيقة لمعدل الإطارات المتغير (VFR)**: يستخرج الطوابع الزمنية للعرض (PTS) الحقيقية مباشرة من حزم الفيديو لمنع عدم تزامن الصوت/الفيديو، مما يعالج مقاطع اللعب ذات معدل الإطارات المتغير (VFR) بسلاسة.
- تسجيل الحركة للصوت + الفيديو:
  - تصنيف مدمج بأوزان قابلة للتعديل (الافتراضي: الصوت 0.6، الفيديو 0.4).
- يتم تصنيف المشاهد بناءً على درجة الحركة المدمجة بدلاً من المدة.
- **التقطيع الذكي للمشاهد**:
  - يفضل تحديد المشاهد الكاملة إذا كانت تتناسب مع الحد الزمني.
  - **حشو المشهد (Padding)**: يضيف مساحة تخزين مؤقتة مدتها 1.5 ثانية إلى نهاية المشاهد لالتقاط الرسوم المتحركة للخروج وتأثيرات التلاشي.
  - **القص الذكي**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (انخفاض الصوت/الحركة) لقصها، وتجنب النهايات المفاجئة.
- اقتصاص ذكي مع خلفية معتمة اختيارية للقطات غير الرأسية.
- منطق إعادة المحاولة أثناء التصيير لتجنب حالات الفشل العابرة.
- التهيئة عبر متغيرات البيئة في ملف `.env`.

## 📋 المتطلبات

- **وحدة معالجة رسومات NVIDIA (GPU)** تدعم CUDA.
- **تعريفات NVIDIA** (يوصى بأن تكون متوافقة مع CUDA 13.0+).
- Python 3.12+
- FFmpeg (يُستخدم لاستخراج الصوت وتشفير NVENC).
- مكتبات النظام: `libgl1`, `libglib2.0-0` (غالبًا ما تكون مطلوبة لمكتبات الرؤية).

اعتماديات Python (انظر `pyproject.toml`):
- `torch`, `torchaudio` (مع دعم CUDA)
- `PyNvCodec`, `PytorchNvCodec` (إطار عمل معالجة الفيديو)

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

إذا واجهت مشاكل تتعلق بعدم عثور PyTorch على وحدة معالجة الرسومات، فارجع إلى دليل التثبيت الخاص به لإصدار CUDA المحدد لديك.

## 💡 الاستخدام

1. ضع الفيديوهات المصدرية داخل مجلد `gameplay/`.
2. قم بتشغيل أداة واجهة سطر الأوامر:

```bash
shorts-maker process
```

يمكنك اختياريًا تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يعرض السجل درجة الحركة لكل مشهد مدمج والقائمة النهائية مرتبة بناءً على هذه الدرجة. يتم تصيير المشاهد الأعلى (حسب كثافة الحركة) أولاً باستخدام NVENC.

## 🐳 Docker (مستحسن)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع أداة NVIDIA Container Toolkit.

**المتطلبات الأساسية**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا تعطل البناء مع خطأ "Segmentation fault" أو خطأ في الذاكرة، قم بتقييد أنوية وحدة المعالجة المركزية (CPU) باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

```bash
docker build -t shorts-maker .

# تشغيل مع صلاحية الوصول إلى GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

لاحظ علامة `--gpus all`، وهي ضرورية ليتمكن التطبيق من الوصول إلى تسريع العتاد.

## ⚙️ الإعدادات (Configuration)

انسخ `.env.example` إلى `.env` وقم بتعديل القيم حسب الحاجة.

المتغيرات المدعومة (القيم الافتراضية معروضة):
- `TARGET_RATIO_W=9` — جزء العرض لنسبة العرض إلى الارتفاع المستهدفة (مثل 9 لنسبة 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع لنسبة العرض إلى الارتفاع المستهدفة (مثل 16 لنسبة 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد المشاهد العلوية التي يتم تصييرها لكل فيديو مصدري.
- `SCENE_THRESHOLD=45.0` — العتبة (Threshold) لاقتطاعات اكتشاف المشهد.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في النطاق [0.0, 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في النطاق [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — أقصى عمق لإعادة المحاولة في حالة فشل التصيير.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول الفيديو القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول الفيديو القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المدمج (بالثواني).
- `SKIP_FIRST_SECONDS=0.0` — عدد الثواني التي يجب تخطيها من بداية الفيديو (مفيد لتخطي شاشات المقدمة).
- `SAVE_FFMPEG_LOGS=False` — ما إذا كان سيتم حفظ سجلات FFmpeg أثناء التصيير.
- `LOG_LEVEL=WARNING` — مستوى السجلات (مثل INFO، DEBUG، WARNING).

## 🛠️ التطوير

### فحص الكود (Linting)

يستخدم هذا المشروع `ruff` لفحص الكود بسرعة.

```bash
pip install ruff
ruff check .
```

## 🧪 تشغيل الاختبارات

توجد اختبارات الوحدة (Unit tests) في مجلد `tests/`. قم بتشغيلها باستخدام:

```bash
pytest -q
```

ملاحظة: تم تصميم الاختبارات لمحاكاة (mock) توافر وحدة معالجة الرسومات (GPU) في حال عدم وجودها، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **"internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA ثقيلة (مثل VPF) باستخدام جميع أنوية وحدة المعالجة المركزية (CPU) المتاحة. لإصلاح ذلك، قم بتقييد عدد الأنوية المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(كبديل، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) لـ Docker/WSL2 في إعدادات نظامك).*
- **"WSL integration with distro unexpectedly stopped" / نفاد الذاكرة (OOM) أثناء `docker run`**: يمكن أن تستهلك معالجة الفيديو عالي الدقة مقدارًا كبيرًا من ذاكرة RAM/VRAM، مما يتسبب في تعطل الجهاز الظاهري لـ WSL2 بسبب خطأ نفاد الذاكرة. لإصلاح ذلك، قم بتقييد عدد أنوية وحدة المعالجة المركزية (CPU) التي يمكن للحاوية (container) استخدامها أثناء التنفيذ عن طريق إضافة علامة `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker باستخدام `--gpus all` أو أن لديك مجموعة أدوات CUDA الصحيحة مثبتة محليًا.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، يحاول السكربت الرجوع إلى التشفير البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات لديك تدعم NVENC وما إذا كانت التعريفات محدثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).