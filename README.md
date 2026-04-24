# LTX-2.3 RunPod Serverless Service

Standalone deployment package for a RunPod serverless endpoint that serves
[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) for three inference modes:

| Mode | Pipeline class | Inputs |
|---|---|---|
| `i2v` | `TI2VidTwoStagesPipeline` | prompt + reference image |
| `a2v` | `A2VidPipelineTwoStage` | prompt + reference image + audio |
| `distilled` | `DistilledPipeline` | prompt + reference image (fast, no guider) |

Output: MP4 uploaded to Cloudflare R2, returned as a presigned URL (default
24 h TTL) with stage-level timings.

> This folder is **self-contained** and has no runtime coupling to the rest of
> the Nanopapaya repo. Nothing in `src/`, `drizzle/`, or `MODEL_REGISTRY.md`
> is modified by deploying it.

## Prerequisites

- A RunPod account with serverless + network volumes enabled.
- A HuggingFace account with a token that has access to
  `google/gemma-3-12b-it-qat-q4_0-unquantized` (gated) and `Lightricks/LTX-2.3`.
- A Cloudflare R2 bucket + API token (write scope).
- A container registry the RunPod account can pull from.

## One-time setup

### 1. Create a network volume and download weights

1. In the RunPod console, create a **Network Volume** (~200 GB, same region as
   your serverless workers will live in).
2. Spin up a cheap GPU or CPU pod (any image with `huggingface_hub` installed)
   and mount the volume at `/workspace`.
3. Inside that pod:

   ```bash
   pip install -U huggingface_hub
   export HF_TOKEN=<your-hf-token>
   bash scripts/download_weights.sh
   bash scripts/verify_weights.sh
   ```

   The verify script lists expected vs actual sizes and exits non-zero on any
   missing / truncated asset.

   Files that end up on the volume:

   ```
   /workspace/models/ltx-2.3-22b-dev.safetensors
   /workspace/models/ltx-2.3-22b-distilled.safetensors
   /workspace/models/ltx-2.3-22b-distilled-lora-384.safetensors
   /workspace/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
   /workspace/models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors
   /workspace/models/gemma-3-12b/...
   ```

4. Terminate the download pod; keep the volume.

### 2. Pin the LTX-2 commit and build the image

The Dockerfile clones `https://github.com/Lightricks/LTX-2` at a pinned SHA.

```bash
# from this folder
git ls-remote https://github.com/Lightricks/LTX-2 HEAD        # grab the 40-char SHA
# edit Dockerfile -> replace <PIN_SHA> with it

# verify the base image tag is still published on Docker Hub:
#   https://hub.docker.com/r/runpod/pytorch/tags
# if a newer PyTorch 2.7 / CUDA 12.x tag exists, bump LTX_BASE_IMAGE accordingly.

docker build -t <registry>/ltx-video:<tag> .
docker push <registry>/ltx-video:<tag>
```

### 3. Create the serverless endpoint

Open the RunPod console and follow every setting in
[`endpoint-config.md`](./endpoint-config.md). Attach the network volume at
`/workspace`, paste the env vars from `.env.example`, and save.

## Input schema

```jsonc
{
  "mode": "i2v" | "a2v" | "distilled",   // required
  "prompt": "string",                     // required, non-empty
  "negative_prompt": "string",            // optional, default built-in
  "image_url": "string",                  // required for i2v / a2v / distilled
  "audio_url": "string",                  // required for a2v
  "width": 768,                           // default 768, MUST be divisible by 32
  "height": 1344,                         // default 1344, MUST be divisible by 32
  "num_frames": 121,                      // default 121, MUST satisfy (8k+1), k>=1
  "frame_rate": 25.0,                     // default 25
  "num_inference_steps": 40,              // default 40, ignored by distilled
  "seed": 42,                             // optional, random if omitted
  "image_strength": 1.0,                  // reference-image influence
  "audio_start_time": 0.0,                // a2v only
  "audio_max_duration": null,             // a2v only
  "enhance_prompt": true                  // gemma-powered prompt enhancement
}
```

## Output schema

```jsonc
{
  "url": "https://...mp4",              // presigned GET URL (or R2_PUBLIC_BASE/key)
  "mode": "i2v",
  "width": 768, "height": 1344,
  "num_frames": 121, "frame_rate": 25.0,
  "duration_s": 4.84,
  "file_size_bytes": 12345678,
  "timings": { "pipeline_s": 71.3, "encode_s": 1.2, "upload_s": 0.9 }
}
```

On failure:

```jsonc
{
  "error_type":    "validation" | "pipeline" | "upload",
  "error_message": "…",
  "stage_failed":  null | "stage1" | "stage2" | "encode" | "upload"
}
```

## Local smoke test (optional)

On a dev box with the network volume mounted and the LTX-2 venv active:

```bash
export LTX_WEIGHTS_ROOT=/workspace/models
export LOCAL_TEST_IMAGE=/abs/path/to/ref.png
export LOCAL_TEST_AUDIO=/abs/path/to/voice.wav
python test/local_test.py        # writes outputs/*.mp4
```

R2 + network fetches are monkey-patched away — outputs land in `./outputs/`.

## End-to-end benchmark

Against a live endpoint:

```bash
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_API_KEY=...
export BENCH_IMAGE_URL=https://.../ref.png
python test/benchmark.py
```

Reports success rate + p50/p95/p99 + rough H100-PCIe cost per run.

## Troubleshooting

- **OOM during stage 1.** Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is set (baked into the Dockerfile ENV). Double-check FP8 quantisation is
  enabled — the handler always calls `QuantizationPolicy.fp8_cast()`.
- **`ValueError: num_frames must satisfy (8k+1)`.** The pipelines require the
  frame count to be 1 mod 8. Valid values: 9, 17, 25, 33, 41, 49, …, 121, 241.
- **`ValueError: must be a positive multiple of 32`.** Width/height must be
  divisible by 32. Typical portrait: 768×1344. Square: 768×768 or 1024×1024.
- **Cold start > 4 min.** First load quantises the 22 B checkpoint to FP8 on
  device. Subsequent invocations within the idle window are warm. Consider
  setting `LTX_DEFAULT_PIPELINE=i2v` (or whichever mode dominates traffic) so
  FlashBoot preloads it.
- **Flash-Attention wheels fail on non-Hopper GPUs.** Flash-Attention 3 is
  Hopper-only; on A100 / L40 the install still succeeds but the code path
  falls back to the stock attention kernel at a speed cost.
- **`Audio object did not expose .path / .save / .waveform`.** The installed
  LTX-2 revision has changed the Audio class. Update `_extract_audio_wav` in
  `handler.py` — search for `UPSTREAM-CONFLICT` to find the adapter.

## Files in this folder

| File | Purpose |
|---|---|
| `handler.py` | RunPod serverless entrypoint |
| `Dockerfile` | Image definition (PyTorch 2.7 / CUDA 12.8 base) |
| `.env.example` | Full env-var reference |
| `.dockerignore` | Excludes test outputs / venvs from the build context |
| `scripts/download_weights.sh` | Populates the network volume (one-time) |
| `scripts/verify_weights.sh` | Sanity-checks volume contents |
| `test/local_test.py` | In-process smoke test with mocked R2 |
| `test/benchmark.py` | p50/p95/p99 benchmark against a live endpoint |
| `endpoint-config.md` | RunPod console checklist |
