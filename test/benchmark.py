"""Benchmarks a deployed LTX-2.3 RunPod serverless endpoint.

Submits 10 varied I2V prompts and reports p50/p95/p99 latency, success rate,
and a rough H100-PCIe cost estimate.

Required env:
  RUNPOD_ENDPOINT_ID
  RUNPOD_API_KEY
  BENCH_IMAGE_URL   public URL to a reference image usable by the endpoint
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import runpod

# Approximate on-demand H100 PCIe serverless price. Adjust if RunPod updates pricing.
H100_PCIE_USD_PER_SECOND = 0.00076

PROMPTS = [
    "A person speaking directly to the camera, soft key light, subtle head motion",
    "An energetic street performer juggling at golden hour, shallow depth of field",
    "A luxury watch rotating on a matte black pedestal, studio rim light",
    "Cinematic close-up of coffee pouring into a ceramic mug, steam rising",
    "A glass skyscraper reflecting sunset clouds, slow crane up",
    "Hands typing on a mechanical keyboard, neon purple backlight",
    "A chef searing a steak on a hot pan, sparks flying, slow motion",
    "A stylized 3D character waving, pastel gradient background",
    "A drone shot sweeping over a mountain lake at dawn, mist on water",
    "A product unboxing, minimalist white desk, overhead angle",
]


def run_one(endpoint: runpod.Endpoint, prompt: str, image_url: str) -> tuple[bool, float]:
    t0 = time.monotonic()
    try:
        job = endpoint.run_sync(
            {
                "input": {
                    "mode": "i2v",
                    "prompt": prompt,
                    "image_url": image_url,
                    "width": 768,
                    "height": 1344,
                    "num_frames": 121,
                    "frame_rate": 25.0,
                    "num_inference_steps": 30,
                    "seed": 42,
                }
            },
            timeout=900,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return False, time.monotonic() - t0
    elapsed = time.monotonic() - t0
    ok = isinstance(job, dict) and "url" in job and "error_type" not in job
    if not ok:
        print(f"ERROR: {job}", file=sys.stderr)
    return ok, elapsed


def main() -> int:
    endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    image_url = os.environ["BENCH_IMAGE_URL"]

    endpoint = runpod.Endpoint(endpoint_id)

    successes: list[float] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(run_one, endpoint, p, image_url) for p in PROMPTS]
        for fut in as_completed(futures):
            ok, seconds = fut.result()
            if ok:
                successes.append(seconds)
            else:
                failures += 1
            print(f"  {'ok' if ok else 'FAIL'}  {seconds:.1f}s")

    if not successes:
        print("No successful runs.")
        return 1

    p50 = statistics.median(successes)
    p95 = statistics.quantiles(successes, n=20)[-1] if len(successes) >= 2 else successes[0]
    p99 = max(successes)
    avg = sum(successes) / len(successes)
    total = len(successes) + failures
    rate = len(successes) / total

    print("\n=== summary ===")
    print(f"runs:         {total}  (ok={len(successes)} fail={failures})")
    print(f"success rate: {rate:.0%}")
    print(f"p50:          {p50:.1f}s")
    print(f"p95:          {p95:.1f}s")
    print(f"p99 (max):    {p99:.1f}s")
    print(f"avg cost:     ${avg * H100_PCIE_USD_PER_SECOND:.3f} per run (H100 PCIe est.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
