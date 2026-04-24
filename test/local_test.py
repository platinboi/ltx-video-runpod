"""Exercises handler.handler() in-process with R2 upload mocked.

Run on an H100 dev box that has the LTX-2 venv active and the network volume
mounted at /workspace. Outputs are written to ./outputs/<mode>.mp4.

Required env:
  LTX_WEIGHTS_ROOT    (defaults to /workspace/models)
  LOCAL_TEST_IMAGE    path to a reference image (any size; script resizes URL)
  LOCAL_TEST_AUDIO    path to a wav/mp3 for the a2v test
"""
from __future__ import annotations

import os
import sys
import time
import shutil
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Make the local file paths addressable as file:// URLs.
image_local = Path(os.environ["LOCAL_TEST_IMAGE"]).resolve()
audio_local = Path(os.environ["LOCAL_TEST_AUDIO"]).resolve()
if not image_local.exists() or not audio_local.exists():
    sys.exit(f"missing fixture: image={image_local} audio={audio_local}")

# Stub R2 env so handler import doesn't complain during lazy construction.
os.environ.setdefault("R2_ACCOUNT_ID", "stub")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")
os.environ.setdefault("R2_BUCKET", "stub")

sys.path.insert(0, str(HERE.parent))
import handler  # noqa: E402

# Stub network fetchers and R2 upload.
def fake_download(url: str, suffix: str, work):
    src = Path(url.replace("file://", ""))
    dst = Path(work) / f"in{suffix}"
    shutil.copy(src, dst)
    return dst

def fake_upload(local, key, attempts=3):
    dst = OUTPUT_DIR / Path(key).name
    shutil.copy(local, dst)
    return f"file://{dst}"

CASES = [
    {
        "name": "i2v_portrait_5s",
        "payload": {
            "mode": "i2v",
            "prompt": "A person speaking directly to the camera, soft key light, subtle head motion",
            "image_url": f"file://{image_local}",
            "width": 768,
            "height": 1344,
            "num_frames": 121,
            "frame_rate": 25.0,
            "num_inference_steps": 30,
            "seed": 42,
        },
    },
    {
        "name": "a2v_10s",
        "payload": {
            "mode": "a2v",
            "prompt": "A talking-head avatar, natural lip sync, studio lighting",
            "image_url": f"file://{image_local}",
            "audio_url": f"file://{audio_local}",
            "width": 768,
            "height": 1344,
            "num_frames": 241,
            "frame_rate": 24.0,
            "num_inference_steps": 30,
            "seed": 42,
        },
    },
    {
        "name": "distilled_5s",
        "payload": {
            "mode": "distilled",
            "prompt": "Cinematic product shot, slow dolly push-in",
            "image_url": f"file://{image_local}",
            "width": 1024,
            "height": 576,
            "num_frames": 121,
            "frame_rate": 25.0,
            "seed": 42,
        },
    },
]


def main() -> int:
    with patch.object(handler, "_download", side_effect=fake_download), \
         patch.object(handler, "_upload_r2", side_effect=fake_upload):
        for case in CASES:
            name = case["name"]
            print(f"\n=== {name} ===")
            t0 = time.monotonic()
            result = handler.handler({"id": name, "input": case["payload"]})
            elapsed = time.monotonic() - t0
            if "error_type" in result:
                print(f"FAIL {name}: {result}")
                continue
            size = result["file_size_bytes"]
            print(f"OK   {name}: {size/1_048_576:.1f} MiB in {elapsed:.1f}s  timings={result['timings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
