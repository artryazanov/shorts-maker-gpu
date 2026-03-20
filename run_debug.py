import sys
from pathlib import Path
from unittest import mock
import tempfile

sys.path.append(str(Path(__file__).resolve().parent))
import tests.test_render as tr

print("Starting debug run...")
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    try:
        tr.test_render_video_gpu(tmp_path=tmp_path)
    except Exception as e:
        print("Crashed:", e)
print("Finished!")
