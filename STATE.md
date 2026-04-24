# LTX-2.3 RunPod — Deployment State

Last verified: **2026-04-24**

## What's deployed

| Resource | ID / Value |
|---|---|
| RunPod endpoint | `v2vdehrn1v3y38` (name: `ltx-video-runpod`) |
| Template | `t8adh8yldl` |
| Registry image tag | `registry.runpod.net/platinboi-ltx-video-runpod-main-dockerfile:<sha7>` |
| Network volume | `bya0i5v4mg` (name: `ltx-weights`, 200 GB, EUR-IS-1) |
| GPUs (fallback order) | H100 PCIe → H100 SXM → H100 NVL |
| Workers (min / max) | 0 / 2 |
| Idle timeout | 300 s |
| Execution timeout | 600 s |
| FlashBoot | on |
| Scaler | QUEUE_DELAY, value 4 |
| Region | EUR-IS-1 (forced by volume) |

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
curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/v2vdehrn1v3y38" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"workersMax":0}'
sleep 8
curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/v2vdehrn1v3y38" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"workersMax":2}'
```

## Env vars (on the TEMPLATE, not the endpoint)

PATCHing `env` on `/v1/templates/<id>` **replaces** the whole map — always ship the full set:

```
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET=nanopapaya
R2_PUBLIC_BASE=https://r2.nanopapaya.io
LTX_WEIGHTS_ROOT=/runpod-volume/models
LTX_SIGNED_URL_TTL=86400
LTX_DISTILLED_LORA_STRENGTH=0.6
```

> `LTX_WEIGHTS_ROOT` must point at `/runpod-volume/models`, not `/workspace/models`. Template's `volumeMountPath` field is advisory — serverless actually mounts at `/runpod-volume`. `handler._resolve_weights_root()` auto-detects, so either value works, but the env should reflect reality.

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
- `uv sync --frozen --extra xformers` re-downloads ~3 GB wheels, ~55 s.

Cold full build: **6–8 min**. Near-warm (same code): **4–5 min**.

Build steps are ordered for minimal rebuild scope:
1. `apt-get install` (stable)
2. `curl uv install`
3. `git clone LTX-2 @ pinned SHA`
4. `uv sync --frozen --extra xformers`
5. `uv pip install` serverless deps
6. Build-time smoke test (imports)
7. `COPY handler.py` (rebuilds on every handler change)
8. CMD

If caching becomes painful, switch to GHA + GHCR with `cache-from: type=gha,mode=max` — precedent in `runpod-ffmpeg-suite` repo. Currently out of scope (user rejected GHA).

## Known issues / TODO

- [ ] Verify I/O quality on the distilled mode end-to-end (test pending as of this doc).
- [ ] `LTX_DEFAULT_PIPELINE` is unset → each mode cold-loads on first call. Consider setting to `i2v` once we know the hot path.
- [ ] Flash-Attention 3 wheels install via `--extra xformers`; only accelerates on Hopper (H100/H200). If we add A100/L40 fallback GPUs later, confirm the attention fallback works or pin FA to a compatible variant.

## Session history (what already went wrong once)

In order, during the initial bring-up:
1. Fabricated base image tag — real format is `runpod/pytorch:<img>-cu<cuda>-torch<torch>-ubuntu<ver>`.
2. Assumed `uv sync` creates `.venv` — it doesn't under the base image's env.
3. Missed upstream circular import; smoke test caught it on the second pass.
4. Assumed `volumeMountPath: /workspace` meant workers mount at `/workspace`; serverless uses `/runpod-volume`.
5. Used stale `ImageConditioningInput` field names.

Every one of these can regress on an LTX-2 bump or a base-image refresh — keep the build-time smoke test in place.
