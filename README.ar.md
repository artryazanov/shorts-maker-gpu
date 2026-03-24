> 🌐 **اللغات:** [English](README.md) | [Русский](README.ru.md) | [ไทย](README.th.md) | [中文](README.zh.md) | [Español](README.es.md) | [العربية](README.ar.md)

# 🎬 Shorts Maker (مُحسّن لمعالجات الرسومات GPU)

يقوم Shorts Maker بإنشاء مقاطع فيديو رأسية قصيرة من لقطات اللعب الطويلة. تقوم مكتبة وأداة سطر الأوامر (CLI) المكتوبة بلغة Python هذه باكتشاف المشاهد، وحساب ملفات تعريف الحركة الصوتية والمرئية (شدة الصوت + الحركة المرئية)، وتجمعها لترتيب المشاهد حسب الكثافة الإجمالية. ثم تقوم بقص الفيديو إلى نسبة العرض إلى الارتفاع المطلوبة وعرض مقاطع قصيرة جاهزة للرفع.

**تم تحسين هذه النسخة بشكل كبير لمعالجات رسومات NVIDIA باستخدام CUDA.**

للحصول على النسخة الأصلية التي تعتمد على المعالج المركزي (CPU) فقط، يُرجى زيارة [Shorts Maker](https://github.com/artryazanov/shorts-maker).

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

- **معالجة مُسرّعة بواسطة معالج الرسومات (GPU)**:
  - **فك التشفير وتغيير الحجم باستخدام الأجهزة (Hardware)**: دمج أصلي لإطار عمل معالجة الفيديو من NVIDIA (VPF) عبر `PyNvCodec`. يقوم بفك التشفير وتغيير الحجم وتحويل مساحات الألوان مباشرة على NVDEC.
  - **اكتشاف المشاهد**: تنفيذ مخصص باستخدام VPF و OpenCV.
  - **تحليل الصوت**: يستخدم `torchaudio` على وحدة المعالجة الرسومية لحساب سريع لـ RMS والتدفق الطيفي.
  - **تحليل الفيديو**: تدفق ذاكرة GPU بدون نسخ (Zero-copy) لتقدير ثابت للحركة (يستبدل فهارس الإطارات الثقيلة).
  - **معالجة الصور**: عوامل تشغيل PyTorch الأصلية تُستخدم للعمليات الثقيلة مثل تمويه الخلفيات (الالتفافات القابلة للفصل).
  - **العرض (Rendering)**: محرك مخصص من PyTorch+NVENC للحصول على أداء عرض عالٍ (تمت إزالة MoviePy من مسار العرض).
  - **معالجة دفعية قوية**: تعمل معالجة الفيديو في عمليات فرعية معزولة تمامًا، وتمسح سياقات CUDA بالكامل بين الملفات لمنع تجزئة ذاكرة الوصول العشوائي للفيديو (VRAM) وتجنب أعطال نفاد الذاكرة (OOM) (خاصة في Docker/WSL).
- تسجيل الحركة للصوت والفيديو:
  - تصنيف مدمج بأوزان قابلة للتعديل (الافتراضي: الصوت 0.6، الفيديو 0.4).
- تُصنف المشاهد بناءً على نتيجة الحركة المدمجة بدلاً من المدة.
- **قص ذكي للمشاهد**:
  - يُفضل اختيار المشاهد المكتملة إذا كانت تتناسب مع الحد الزمني.
  - **حشو المشهد**: يضيف مخزنًا مؤقتًا مدته 1.5 ثانية في نهاية المشاهد لالتقاط الرسوم المتحركة للخروج وتأثيرات التلاشي.
  - **اقتطاع ذكي**: بالنسبة للمشاهد الطويلة، يبحث عن اللحظات "الهادئة" (انخفاض مستوى الصوت/الحركة) لقصها، وتجنب النهايات المفاجئة.
- قص ذكي مع إمكانية تمويه الخلفية (اختياري) للقطات غير الرأسية.
- منطق إعادة المحاولة أثناء العرض لتجنب الإخفاقات الزائفة.
- الضبط والتكوين عبر متغيرات البيئة في ملف `.env`.

## 📋 المتطلبات

- **وحدة معالجة رسومات NVIDIA (GPU)** تدعم CUDA.
- **تعريفات NVIDIA** (يُوصى بتعريفات متوافقة مع CUDA الإصدار 13.0 أو أحدث).
- Python 3.12 أو أحدث
- FFmpeg (مُستخدم لاستخراج الصوت وتشفير NVENC).
- مكتبات النظام: `libgl1`، `libglib2.0-0` (غالبًا ما تكون مطلوبة لمكتبات الرؤية).

اعتماديات Python (انظر `pyproject.toml`):
- `torch`، `torchaudio` (مع دعم CUDA)
- `PyNvCodec`، `PytorchNvCodec` (إطار عمل معالجة الفيديو)

## 🚀 التثبيت

### عبر PyPI (موصى به)

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

# قم بتثبيت المكتبة والاعتماديات الخاصة بها
pip install -e .
```

إذا واجهت مشاكل في عدم عثور PyTorch على وحدة المعالجة الرسومية (GPU)، فارجع إلى دليل التثبيت الخاص به لإصدار CUDA المحدد لديك.

## 💡 الاستخدام

1. ضع مقاطع الفيديو المصدر داخل مجلد `gameplay/`.
2. قم بتشغيل أداة سطر الأوامر (CLI):

```bash
shorts-maker process
```

يمكنك اختياريًا تخصيص مجلدات الإدخال والإخراج والحدود القصوى للمشاهد:
```bash
shorts-maker process --input-dir my_videos/ --output-dir my_shorts/ --scene-limit 3
```

3. يتم حفظ المقاطع المُنشأة في مجلد `generated/`.

أثناء المعالجة، يُظهر السجل (log) نتيجة حركة لكل مشهد مدمج والقائمة النهائية مرتبة حسب تلك النتيجة. يتم عرض (rendering) أفضل المشاهد (من حيث كثافة الحركة) أولاً باستخدام NVENC.

## 🐳 Docker (موصى به)

أسهل طريقة لتشغيل هذا التطبيق هي استخدام Docker مع مجموعة أدوات NVIDIA Container.

**متطلب مسبق**: يجب تثبيت [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) على الجهاز المضيف.

البناء والتشغيل:

*(ملاحظة: إذا تعطل البناء برسالة "Segmentation fault" أو خطأ في الذاكرة، فقم بالحد من أنوية المعالج المستخدمة باستخدام `docker build --cpuset-cpus="0,1" -t shorts-maker .` بدلاً من ذلك).*

```bash
docker build -t shorts-maker .

# التشغيل مع منح الصلاحية للوصول إلى GPU
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

لاحظ العلامة `--gpus all`، وهي ضرورية لتمكين التطبيق من الوصول إلى تسريع الأجهزة (Hardware acceleration).

## ⚙️ الإعدادات (Configuration)

انسخ `.env.example` إلى `.env` وقم بتعديل القيم حسب الحاجة.

المتغيرات المدعومة (القيم الافتراضية معروضة):
- `TARGET_RATIO_W=9` — جزء العرض من نسبة العرض إلى الارتفاع المستهدفة (مثال: 9 للنسبة 9:16).
- `TARGET_RATIO_H=16` — جزء الارتفاع من نسبة العرض إلى الارتفاع المستهدفة (مثال: 16 للنسبة 9:16).
- `SCENE_LIMIT=4` — الحد الأقصى لعدد أفضل المشاهد التي يتم عرضها لكل فيديو مصدر.
- `SCENE_THRESHOLD=45.0` — حد القطع لاكتشاف المشاهد.
- `X_CENTER=0.5` — مركز القص الأفقي في النطاق [0.0, 1.0].
- `Y_CENTER=0.5` — مركز القص العمودي في النطاق [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — الحد الأقصى لعدد محاولات إعادة العرض (rendering) في حال الفشل.
- `MIN_SHORT_LENGTH=15` — الحد الأدنى لطول المقطع القصير بالثواني.
- `MAX_SHORT_LENGTH=179` — الحد الأقصى لطول المقطع القصير بالثواني.
- `MAX_COMBINED_SCENE_LENGTH=300` — الحد الأقصى للطول المدمج (بالثواني).
- `SAVE_FFMPEG_LOGS=False` — تحديد ما إذا كان يجب حفظ سجلات FFmpeg أثناء العرض.
- `LOG_LEVEL=WARNING` — مستوى تسجيل الأحداث (مثال: INFO, DEBUG, WARNING).

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

ملاحظة: تم تصميم الاختبارات لمحاكاة (Mock) توفر وحدة المعالجة الرسومية (GPU) في حال غيابها، بحيث يمكن تشغيلها في بيئات التكامل المستمر (CI) القياسية.

## 🚑 استكشاف الأخطاء وإصلاحها

- **"internal compiler error: Segmentation fault" أثناء `docker build`**: يحدث هذا عادةً بسبب خطأ نفاد الذاكرة (OOM) عندما يحاول Docker تجميع مكتبات C++/CUDA الثقيلة (مثل VPF) باستخدام جميع أنوية المعالج المتاحة. لإصلاح ذلك، قم بتحديد عدد أنوية المعالج المستخدمة أثناء عملية البناء:
  ```bash
  docker build --cpuset-cpus="0,1" -t shorts-maker .
  ```
  *(كبديل، يمكنك زيادة حد ذاكرة الوصول العشوائي (RAM) المخصصة لـ Docker/WSL2 في إعدادات نظامك).*
- **"Torch not installed" / "CUDA not available"**: تأكد من أنك تقوم بالتشغيل داخل حاوية Docker مع علامة `--gpus all` أو أن لديك مجموعة أدوات CUDA الصحيحة مثبتة محليًا.
- **خطأ NVENC**: إذا فشل `h264_nvenc`، سيحاول البرنامج النصي العودة إلى التشفير البرمجي (`libx264`). تحقق مما إذا كانت وحدة المعالجة الرسومية لديك تدعم NVENC وما إذا كانت التعريفات مُحدثة.

## 📄 الترخيص

تم إصدار هذا المشروع بموجب [ترخيص MIT](LICENSE).