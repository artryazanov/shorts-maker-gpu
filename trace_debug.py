import sys
import threading
import traceback
from pathlib import Path
from unittest import mock
import tempfile

sys.path.append(str(Path(__file__).resolve().parent))
import tests.test_render as tr

def dump_stack():
    print("\n--- TRACE DUMP ---")
    for thread_id, frame in sys._current_frames().items():
        print(f"Thread {thread_id}:")
        traceback.print_stack(frame)
    print("------------------\n")
    import os
    os._exit(1)

timer = threading.Timer(5.0, dump_stack)
timer.daemon = True
timer.start()

print("Starting debug run...")
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    try:
        tr.test_render_video_gpu(tmp_path=tmp_path)
    except Exception as e:
        print("Crashed:", e)
print("Finished!")
timer.cancel()
