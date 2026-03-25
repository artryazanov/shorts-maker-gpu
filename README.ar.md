> 🌐 **Languages:** [English](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.md) | [Русский](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ru.md) | [ไทย](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.th.md) | [中文](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.zh.md) | [Español](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.es.md) | [العربية](https://github.com/artryazanov/shorts-maker-gpu/blob/main/README.ar.md)

# 🎬 صانع المقاطع القصيرة (محسّن لوحدات معالجة الرسومات GPU)

يقوم صانع المقاطع القصيرة بإنشاء مقاطع فيديو رأسية من لقطات ألعاب الفيديو الطويلة. تقوم مكتبة وأداة سطر الأوامر (CLI) المبرمجة بلغة بايثون باكتشاف المشاهد، وحساب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، ودمجها لترتيب المشاهد بناءً على الكثافة الكلية. بعد ذلك، تقوم باقتصاص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وتصدير مقاطع قصيرة جاهزة للرفع.

**تم تحسين هذا الإصدار بشكل كبير لوحدات معالجة الرسومات NVIDIA باستخدام CUDA.**

للحصول على الإصدار الأصلي المخصص لوحدة المعالجة المركزية (CPU) فقط، يرجى زيارة [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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
  - **فك التشفير وتغيير الحجم العتادي**: دمج أصلي لإطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير وتغيير الحجم وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة معالجة الرسومات (GPU) لحساب التدفق الطيفي وجذر متوسط المربع (RMS) السريع.
  - **تحليل الفيديو**: تدفق ذاكرة GPU بدون نسخ لتقدير الحركة المستقر (يستبدل مؤشرات الإطارات الثقيلة).
  - **معالجة الصور**: عوامل PyTorch الأصلية المستخدمة للعمليات الثقيلة مثل تمويه الخلفيات (الالتفافات القابلة للفصل).
  - **التصدير**: محرك PyTorch+NVENC مخصص للتصدير عالي الأداء (تم إزالة MoviePy من مسار التصدير).
  - **معالجة دفعات قوية**: يتم تشغيل معالجة الفيديو في عمليات فرعية معزولة بالكامل، مما يمسح سياقات CUDA تمامًا بين الملفات لمنع تجزئة VRAM وانهيارات نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
- تقييم حركة الصوت + الفيديو:
  - ترتيب مدمج بأوزان قابلة للضبط (الافتراضي: الصوت 0.6، الفيديو 0.4).
- ترتيب المشاهد بناءً على درجة الحركة المدمجة بدلاً من المدة.
- **تقطيع المشاهد الذكي**:
  - يُفضل اختيار المشاهد الكاملة إذا كانت تتناسب مع الحد الزمني.
  - **حشو المشهد**: يضيف مخزنًا مؤقتًا مدته 1.5 ثانية إلى نهاية المشاهد لالتقاط حركات الخروج والتلاشي.
  - **التشذيب الذكي**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (انخفاض الصوت/الحركة) للقص، لتجنب النهايات المفاجئة.
- اقتصاص ذكي مع خلفية مموهة اختيارية للقطات غير الرأسية.
- منطق إعادة المحاولة أثناء التصدير لتجنب الأخطاء الزائفة.
- التكوين عبر متغيرات البيئة `.env`.

## 📋 المتطلبات

- **وحدة معالجة الرسومات NVIDIA** مع دعم CUDA.
- **تعريفات NVIDIA** (يوصى بالإصدارات المتوافقة مع CUDA 13.0+).
- بايثون 3.12+
- FFmpeg (يُستخدم لاستخراج الصوت وتشفير NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالباً ما تكون ضرورية لمكتبات الرؤية).

تبعيات بايثون (راجع `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (مستحسن)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA. ثم قم بتثبيت الحزمة مباشرة:

```bash
pip install shorts-maker-gpu
```

### الإعداد اليدوي من المصدر (لينكس مع CUDA)

تأكد من تثبيت تعريفات NVIDIA ومجموعة أدوات CUDA.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker-gpu
python3 -m venv venv
source venv/bin/activate

# تثبيت المكتبة وتبعياتها
pip install -e .
```

إذا واجهت مشكلات بسبب عدم تمكن PyTorch من العثور على وحدة معالجة الرسومات (GPU)، فارجع إلى دليل التثبيت الخاص به لإصدار CUDA الذي لديك.

## 💡 الاستخدام

1. ضع الفيديوهات المصدرية داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك اختيارياً تخصيص مجلدات الإدخال والإخراج وحدود المشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. تتم كتابة المقاطع التي تم إنشاؤها في مجلد `generated/`.

أثناء المعالجة، يُظهر السجل درجة الحركة لكل مشهد مدمج والقائمة النهائية مرتبة حسب تلك الدرجة. يتم تصدير أفضل المشاهد (حسب كثافة الحركة) أولاً باستخدام NVENC.

## 🐳 Docker (مستحسن)

أسهل طريقة لتشغيل هذا التطبيق هي باستخدام Docker مع مجموعة أدوات الحاويات من NVIDIA (NVIDIA Container Toolkit).

**متطلب أساسي**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا فشلت عملية البناء بـ "Segmentation fault" أو خطأ في الذاكرة، قم بتقليل أنوية المعالج المستخدمة باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

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

لاحظ علامة `--gpus all`، وهي ضرورية لكي يتمكن التطبيق من الوصول إلى تسريع الأجهزة العتادي.

## ⚙️ التكوين

انسخ `.env.example` إلى `.env` واضبط القيم حسب الحاجة.

المتغيرات المدعومة (الافتراضيات المعروضة):
- `TARGET_RATIO_W=9` — جزء العرض من نسبة العرض إلى الارتفاع المستهدفة (مثل 9 لـ 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع من نسبة العرض إلى الارتفاع المستهدفة (مثل 16 لـ 9:16).
- `SCENE_LIMIT=4` — أقصى عدد لأفضل المشاهد التي سيتم تصديرها لكل فيديو مصدر.
- `SCENE_THRESHOLD=45.0` — العتبة لقصاصات اكتشاف المشاهد.
- `X_CENTER=0.5` — مركز الاقتصاص الأفقي في نطاق [0.0, 1.0].
- `Y_CENTER=0.5` — مركز الاقتصاص العمودي في نطاق [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لعمق إعادة المحاولة في حالة فشل التصدير.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول المقطع القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول المقطع القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — ما إذا كان سيتم حفظ سجلات FFmpeg أثناء التصدير.
- `LOG_LEVEL=WARNING` — مستوى التسجيل (مثل INFO، DEBUG، WARNING).

## 🛠️ التطوير

### فحص جودة الكود (Linting)

يستخدم هذا المشروع `ruff` لفحص جودة الكود بشكل سريع.

```bash
pip install ruff
ruff check .
```

## 🧪 تشغيل الاختبارات

توجد اختبارات الوحدة (Unit tests) في مجلد `tests/`. قم بتشغيلها باستخدام:

```bash
pytest -q
```

ملاحظة: صُممت الاختبارات لمحاكاة توافر وحدة معالجة الرسومات (GPU) إذا كانت مفقودة، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **"internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA الثقيلة (مثل VPF) باستخدام جميع أنوية وحدة المعالجة المركزية (CPU) المتاحة. لإصلاح ذلك، قم بتقليل عدد أنوية CPU المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(بدلاً من ذلك، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) لـ Docker/WSL2 في إعدادات نظامك).*
- **"WSL integration with distro unexpectedly stopped" / OOM أثناء `docker run`**: يمكن أن تستهلك معالجة الفيديو عالي الدقة مقدارًا كبيرًا من RAM/VRAM، مما يؤدي إلى تعطل الآلة الافتراضية WSL2 بسبب خطأ نفاد الذاكرة (OOM). لإصلاح ذلك، قم بتقليل عدد أنوية وحدة المعالجة المركزية التي يمكن للحاوية استخدامها أثناء التنفيذ عن طريق إضافة علامة `--cpus`:
  ```bash
  docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
  ```
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker باستخدام `--gpus all` أو أن لديك مجموعة أدوات CUDA الصحيحة مثبتة محليًا.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، سيحاول البرنامج النصي الرجوع إلى التشفير البرمجي (`libx264`). تحقق مما إذا كانت وحدة معالجة الرسومات لديك تدعم NVENC وما إذا كانت التعريفات محدثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).