# LTX-2.3 RunPod — Deployment State

Last verified: **2026-04-25**

## What's deployed

| Resource | Value |
|---|---|
| RunPod endpoint | name: `ltx-video-runpod` (ID stored locally; not in repo) |
| Template | tag pattern: `registry.runpod.net/<owner>-ltx-video-runpod-main-dockerfile:<sha7>` |
| Network volume | `ltx-weights`, 200 GB, region EUR-IS-1 |
| GPUs (current) | true Hopper only — H100 PCIe / SXM / NVL + H200 (forced by FA3 wheel; see GPU constraint section) |
| Workers (min / max) | 0 / 2 |
| Idle timeout | 300 s |
| Execution timeout | 1500 s (25 min) |
| FlashBoot | on |
| Scaler | QUEUE_DELAY, value 4 |
| Region | EUR-IS-1 (forced by volume) |

**Live IDs are NOT in this repo** — see your local `.env.local` or RunPod console for the endpoint, template, and volume IDs.

## Pinned versions

| Pin | Value |
|---|---|
| LTX-2 commit | `41d924371612b692c0fd1e4d9d94c3dfb3c02cb3` (in `Dockerfile` as `LTX_COMMIT_SHA`) |
| Base image | `runpod/pytorch:1.0.3-cu1290-torch271-ubuntu2204` |
| `uv` version | whatever the install script pulls at build time (0.11.7 at last build) |

Bump procedure:
1. `git ls-remote https://github.com/Lightricks/LTX-2 HEAD` → update `LTX_COMMIT_SHA` in `Dockerfile`.
2. Check that `ltx_core.loader` still initializes before `ltx_core.quantization` triggers — see upstream trap #1 below.
3. Confirm `ImageConditioningInput` field names haven't changed again (trap #2).
4. Check Docker Hub tags at https://hub.docker.com/r/runpod/pytorch/tags for a newer base image if needed.

## Deploy flow

1. Push to `main` on `github.com/platinboi/ltx-video-runpod`.
2. RunPod's native GitHub builder rebuilds on their infra (no GHA) and patches the template's `imageName` with the new `<sha7>` tag.
3. Warm workers keep running the **old** image until bounced. Toggle `workersMax` to 0 then back to 2 via REST API to force reload:

```bash
curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/<ENDPOINT_ID>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"workersMax":0}'
sleep 8
curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/<ENDPOINT_ID>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"workersMax":2}'
```

## Env vars (on the TEMPLATE, not the endpoint)

See `.env.example` for the full key list. PATCHing `env` on `/v1/templates/<id>` **replaces** the whole map — always ship the full set, never just the delta. `LTX_WEIGHTS_ROOT` must point at `/runpod-volume/models`, not `/workspace/models`. Template's `volumeMountPath` field is advisory — serverless actually mounts at `/runpod-volume`. `handler._resolve_weights_root()` auto-detects, so either value works, but the env should reflect reality.

## Upstream LTX-2 traps (live at pinned commit)

### 1. Circular import — `loader` must be imported before `quantization`

```
ltx_core.quantization.__init__
  → fp8_cast                              (still loading)
    → ltx_core.loader.module_ops          (triggers loader/__init__)
      → fuse_loras
        → fp8_cast._fused_add_round_launch  ← not yet defined → ImportError
```

Workaround in `handler.py`:

```python
# MUST come first, before any ltx_core.quantization / ltx_pipelines import
from ltx_core.loader import (...)
```

The `Dockerfile` runs the same ordered import as a build-time smoke test. Preserve that ordering. If upstream fixes this, the smoke test still passes — no change needed.

### 2. `ImageConditioningInput` field names

Current: `path`, `frame_idx`, `strength`, `crf` (NamedTuple).
Previously (pre-2026-04): `image_path`, `start_frame`, `strength`, `crf`.
`A2VidPipelineTwoStage` wants `images` as bare tuples `(path, frame_idx, strength)` — no `crf`.

### 3. Pipeline return is `(Iterator[torch.Tensor], Audio)`

Audio object shape is undocumented upstream. `handler._extract_audio_wav` tries `.path` → `.save/.to_file/.write` → `.waveform + .sample_rate` → None. If upstream refactors this, grep for `UPSTREAM-CONFLICT`.

### 4. `DistilledPipeline` has a different __init__ than TI2V/A2V

No `checkpoint_path` / `distilled_lora`. Takes `distilled_checkpoint_path` directly. `__call__` has no guiders, no `num_inference_steps`, no `negative_prompt`. Keep the mode-specific branches in `get_pipeline()` and `_run_pipeline()`.

## Base image trap

`runpod/pytorch:*` images ship with `UV_SYSTEM_PYTHON=1` in the environment. `uv sync` therefore installs into the base image's system Python — **no `.venv/` is created**. Implications:

- Don't reference `/opt/ltx-2/.venv/bin/python` or `/opt/ltx-2/.venv/bin/pip` anywhere.
- Use `uv pip install` for extra deps (targets the same system env).
- CMD uses `uv run python -u handler.py` so `uv` picks the right interpreter at runtime.

## Caching behaviour

RunPod's GitHub builder does **not** persist buildkit layer cache across builds:

- Fresh buildkit container per build (`charming_goldberg`, `quirky_ritchie`, etc — different each time).
- Base image (~5 GB) re-extracts every build, ~200 s.
- `uv sync` re-downloads ~3 GB wheels, ~55 s.
- **Layer + cache export takes 10–20 min** on top of the ~5 min of real work — total ~25–35 min per build.
- The cache export writes to `/runpod-volume/registry.runpod.net/...` on the same network volume as our weights; **it has aborted mid-write with MooseFS I/O errors** (`error writing layer blob: sync failed: input/output error`). When this happens the template `imageName` does not advance — push an empty/no-op commit to retrigger. Transient; not our problem.

Build steps are ordered for minimal rebuild scope:
1. `apt-get install` (stable)
2. `curl uv install`
3. `git clone LTX-2 @ pinned SHA`
4. `uv sync --frozen` (no xformers extra; see GPU section)
5. `uv pip install` serverless deps
6. Build-time smoke test (imports)
7. `COPY handler.py` (rebuilds on every handler change)
8. CMD

If build cost becomes painful, switch to GHA + GHCR with `cache-from: type=gha,mode=max` — precedent in `runpod-ffmpeg-suite` repo. Currently out of scope (user rejected GHA).

## Known issues / TODO

- [ ] First end-to-end successful generation still pending — H200 + Mode A test in progress at last commit.
- [ ] `LTX_DEFAULT_PIPELINE` is unset → each mode cold-loads on first call. Consider setting to `i2v` once we know the hot path.
- [ ] **`handler.get_pipeline()` always passes `quantization=fp8_cast()`.** Combined with `LTX_OFFLOAD_MODE=cpu|disk` this raises `ValueError: quantization is not supported with layer streaming`. Either branch quantization on offload mode, or add a separate `LTX_QUANTIZATION` env. Until done, only `LTX_OFFLOAD_MODE=none` is usable, which forces H200-only.
- [ ] If H200 stock dries up: implement the Mode B path so we can use Blackwell RTX PRO 6000 (currently abundant) at ~5 GB peak VRAM.

## Memory + GPU + attention — three intertwined constraints

There are three knobs that have to be set consistently. Get any one wrong and you crash, OOM, or queue forever.

### 1. Attention backend — current image uses **torch SDPA** (no xformers extra)

The Dockerfile's `uv sync --frozen` does NOT include `--extra xformers`. The xformers extra bakes Flash-Attention 3 wheels that **only run on true Hopper sm_90** (H100 PCIe / SXM / NVL, H200). On anything else (Blackwell RTX PRO 6000, Ampere A100, Ada L40), FA3 kernel launch fails with:

```
CUDA error (.../flash-attention/hopper/flash_fwd_launch_template.h:188): invalid argument
```

Worker enters a crash loop: model loads (~2 min) → denoising starts → FA3 fires → kernel fails → worker restarts → repeat. Job stays `IN_PROGRESS` indefinitely; logs show `Starting Serverless Worker` repeated every ~2 min.

**Trade:** SDPA is ~20–30% slower per matmul than FA3 but runs on any modern arch. We chose portability over throughput because EUR-IS-1 Hopper stock is unreliable.

To re-enable FA3: add `--extra xformers` back to the `uv sync` line **and** restrict `gpuTypeIds` to H100/H200 only.

### 2. Quantization vs. offload — they are MUTUALLY EXCLUSIVE

`QuantizationPolicy.fp8_cast()` is **incompatible with `OffloadMode.CPU` or `OffloadMode.DISK`**. Combining them fails with:

```
ValueError: quantization is not supported with layer streaming
```

You must pick one of these three operating modes:

| Mode | Quantization | Offload | Min VRAM | Speed | Stability |
|---|---|---|---|---|---|
| **A. FP8 + NONE** | `QuantizationPolicy.fp8_cast()` | `OffloadMode.NONE` | ~95 GB (94 too tight, see below) | fastest | best on H200 |
| **B. bf16 + CPU** | `None` | `OffloadMode.CPU` | ~5 GB + activations | medium (PCIe-bound) | works on any 24 GB+ GPU |
| **C. bf16 + DISK** | `None` | `OffloadMode.DISK` | ~5 GB + activations | slowest (re-reads weights every step) | works on tiny GPUs |

`handler.get_pipeline()` reads `LTX_OFFLOAD_MODE` env (`none|cpu|disk`, default `cpu`). FP8 is currently always applied — when `LTX_OFFLOAD_MODE=cpu/disk`, the handler must be patched to skip the `quantization` argument or you'll hit the `ValueError`.

> **Open TODO:** `handler.get_pipeline()` doesn't yet branch on `LTX_OFFLOAD_MODE` to drop quantization for CPU/DISK modes. Today it sets both, which works only for `none`. Either branch quantization on offload mode, or add an explicit `LTX_QUANTIZATION` env switch.

### 3. VRAM math for Mode A

22B FP8 transformer (~12 GB) + Gemma-3-12B encoder (~24 GB at bf16) + VAE + activations + KV cache = **~92 GB allocated peak observed on H100 NVL (94 GB)**. OOM with 2.88 MiB free trying to allocate 128 MiB inside `fp8_cast.py:_upcast_and_round`.

→ **Mode A does NOT fit reliably on H100 NVL (94 GB).** It needs **H200 (141 GB)** for headroom. H100 SXM/PCIe (80 GB) are way too small for Mode A.

→ For H100/A100/Blackwell (≤96 GB), use Mode B (bf16 + CPU offload) once the handler is patched to skip quantization in offload mode.

### Current operating point

- Image: SDPA (no xformers extra), portable across all archs.
- `gpuTypeIds`: H200 + H100 family.
- `LTX_OFFLOAD_MODE=none` (Mode A).
- `executionTimeoutMs`: 1500 s (25 min).
- **In practice this means H200 is the only GPU that reliably works.** H100 NVL OOMs on Mode A. H200 has Medium stock (better than H100 NVL Low).

## Regional GPU starvation

EUR-IS-1 (where the volume lives) typically shows **Low** stock for H100 NVL and **no stock** for H100 SXM/PCIe. Globally H100 SXM is "High" but it's in other regions; the region-locked volume can't follow. Symptoms when starved:

- New jobs sit `IN_QUEUE` indefinitely.
- `/health` shows `workers.throttled > 0` or all worker counters at 0 with `inQueue > 0`.
- No error — RunPod silently waits for capacity.

**Workarounds, in order of effort:**

1. **Broaden `gpuTypeIds`** to non-Hopper (Blackwell RTX PRO 6000, A100). **Will not work** as long as the image is built with `--extra xformers` because of the FA3 crash documented above. To use this path, also drop the xformers extra and rebuild — accept the SDPA throughput hit.
2. **Migrate volume to a less-starved region** (e.g. `US-IL-1`, `US-KS-2`). Cost: re-download 117 GB onto a fresh volume, recreate endpoint, ~1 hr of work.
3. **Buy capacity** — set `workersStandby ≥ 1` to keep one worker warm. Costs while idle, but guarantees instant pickup. Not currently configured (defaults to 0).
4. **Switch provider** (Modal, Replicate, fal.ai). Last resort; full deploy rewrite. Modal has the best Hopper coverage for this workload pattern but requires re-pinning weights into a Modal Volume.

If starvation becomes a chronic problem (multiple test jobs queued >30 min in a single day), pivot to option 2 or 4. Don't keep retrying — bouncing the endpoint doesn't help since RunPod's allocator is already trying.

## Session history (what already went wrong once)

Bring-up issues, in order they were hit:
1. Fabricated base image tag — real format is `runpod/pytorch:<img>-cu<cuda>-torch<torch>-ubuntu<ver>`.
2. Assumed `uv sync` creates `.venv` — it doesn't under the base image's `UV_SYSTEM_PYTHON=1` env.
3. Missed upstream circular import; smoke test caught it on the second pass.
4. Assumed `volumeMountPath: /workspace` meant workers mount at `/workspace`; serverless actually uses `/runpod-volume`.
5. Used stale `ImageConditioningInput` field names (`image_path`/`start_frame` → upstream renamed to `path`/`frame_idx`).
6. Built with `--extra xformers` and broadened GPU pool — Blackwell crash-looped on FA3 (`flash_fwd_launch_template.h:188 invalid argument`). Fix: drop xformers extra, fall back to SDPA.
7. EUR-IS-1 chronic Hopper starvation — jobs sat IN_QUEUE with zero workers, no errors. Fix: pre-allocate Mode B path for non-Hopper, or pay for `workersStandby`.
8. Mode A (FP8 + `OffloadMode.NONE`) OOMed on H100 NVL (94 GB) at 92 GB allocated. Fix: pin H200 (141 GB) for Mode A, or use Mode B.
9. Tried Mode A but with `OffloadMode.CPU` to fix OOM → `ValueError: quantization is not supported with layer streaming`. Lesson: FP8 and offload are mutex.
10. RunPod cache-export to network volume aborted with MooseFS I/O error mid-build — template `imageName` did not advance. Fix: empty-commit retrigger.
11. PATCHing `env` on a template **replaces** the whole map; partial patches wipe other keys. Always ship the full set.
12. Endpoint `executionTimeoutMs` defaulted to 600 s and tripped during cold-start FP8 cast. Fix: bump to 1500 s.

Every one of these can regress on an LTX-2 bump, a base-image refresh, or a RunPod policy change. The Dockerfile build-time import smoke test catches #3, #5; the rest live in the operator's head and in this file.
