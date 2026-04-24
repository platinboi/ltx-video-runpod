# RunPod Serverless — Endpoint Configuration

Follow this checklist in the RunPod console (Serverless → New Endpoint) after
the container image has been pushed to a registry.

## GPU / worker

| Field | Value |
|---|---|
| GPU type | **H100 PCIe** (80 GB) |
| Workers — min | **0** |
| Workers — max | **3** |
| Idle timeout | **30 seconds** |
| Execution timeout | **600 seconds** |
| FlashBoot | **Enabled** |
| Active worker warmup | off (rely on FlashBoot + network-volume weights) |

Hopper (H100/H200) is required to hit the FP8 code path with Flash-Attention
3 wheels installed by the `xformers` extra. Other GPU classes will run but
fall back to a slower attention implementation.

## Container

| Field | Value |
|---|---|
| Container image | `<your-registry>/ltx-video:<tag>` |
| Container disk | 20 GB (just for logs / temp outputs) |
| Exposed HTTP ports | none |
| Start command | *(leave blank — Dockerfile `CMD` runs the handler)* |

## Network volume

Mount the pre-populated weights volume at **`/workspace`**. The handler reads
`${LTX_WEIGHTS_ROOT:-/workspace/models}`, so all six assets downloaded by
`scripts/download_weights.sh` must sit under `/workspace/models/`.

## Environment variables

Paste these in the “Environment Variables” panel. Values come from `.env.example`.

| Key | Required | Notes |
|---|---|---|
| `R2_ACCOUNT_ID` | yes | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | yes | R2 token — write scope on the target bucket |
| `R2_SECRET_ACCESS_KEY` | yes | |
| `R2_BUCKET` | yes | Bucket the MP4s are uploaded to |
| `R2_PUBLIC_BASE` | no | Custom CDN base (e.g. `https://media.example.com`). If unset, a signed R2 URL is returned. |
| `LTX_WEIGHTS_ROOT` | no | Defaults to `/workspace/models`. Only change if you mount the volume elsewhere. |
| `LTX_SIGNED_URL_TTL` | no | Seconds; default 86400. |
| `LTX_DEFAULT_PIPELINE` | no | Set to `i2v`, `a2v`, or `distilled` to preload on cold start. |
| `LTX_DISTILLED_LORA_STRENGTH` | no | Default 0.6. |
| `PYTORCH_CUDA_ALLOC_CONF` | pre-set in Dockerfile | `expandable_segments:True` — do not override. |

## Health check

Submit a small `distilled` job against the endpoint after deploy to confirm
weights mount + model loads:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"mode":"distilled","prompt":"test","image_url":"https://example.com/ref.png","width":768,"height":768,"num_frames":41,"frame_rate":25.0,"seed":1}}'
```

First cold start will take 2–4 minutes (weight load + FP8 cast). Warm requests
should complete in under 90 seconds for 121 frames at 768×1344 on H100 PCIe.
