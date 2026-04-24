"""RunPod serverless handler for LTX-2.3 video generation.

Exposes three inference modes backed by three separate LTX-2 pipelines:
  - i2v        -> TI2VidTwoStagesPipeline        (prompt + reference image)
  - a2v        -> A2VidPipelineTwoStage          (prompt + reference image + audio)
  - distilled  -> DistilledPipeline              (prompt + reference image, fast)

Output MP4s are streamed from the pipeline frame iterator + returned Audio,
muxed via ffmpeg, uploaded to Cloudflare R2, and returned as a presigned URL.

Pipeline signatures were captured from upstream on 2026-04-24. If anything
here diverges from the installed LTX-2 revision, a `# UPSTREAM-CONFLICT:`
comment flags the decision instead of silently adapting.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import boto3
import httpx
import runpod
import torch
from botocore.client import Config as BotoConfig
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# LTX-2 imports. These land in the image via `uv sync` inside /opt/ltx-2.
# ---------------------------------------------------------------------------
from ltx_core.components.guiders import MultiModalGuiderParams  # type: ignore
from ltx_core.loader import (  # type: ignore
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraPathStrengthAndSDOps,
)
from ltx_core.quantization import QuantizationPolicy  # type: ignore
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage  # type: ignore
from ltx_pipelines.distilled import DistilledPipeline  # type: ignore
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # type: ignore
from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

WEIGHTS_ROOT = Path(os.environ.get("LTX_WEIGHTS_ROOT", "/workspace/models"))
DEFAULT_PIPELINE = os.environ.get("LTX_DEFAULT_PIPELINE", "").strip().lower() or None
SIGNED_URL_TTL = int(os.environ.get("LTX_SIGNED_URL_TTL", "86400"))
DISTILLED_LORA_STRENGTH = float(os.environ.get("LTX_DISTILLED_LORA_STRENGTH", "0.6"))

CHECKPOINT_22B = WEIGHTS_ROOT / "ltx-2.3-22b-dev.safetensors"
CHECKPOINT_DISTILLED = WEIGHTS_ROOT / "ltx-2.3-22b-distilled.safetensors"
DISTILLED_LORA = WEIGHTS_ROOT / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER = WEIGHTS_ROOT / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = WEIGHTS_ROOT / "gemma-3-12b"

NEGATIVE_PROMPT_DEFAULT = (
    "worst quality, low quality, blurry, jpeg artifacts, deformed, "
    "disfigured, watermark, text, letters, logo"
)


# ---------------------------------------------------------------------------
# R2 client (lazy)
# ---------------------------------------------------------------------------

_R2_CLIENT: Optional[Any] = None


def _r2() -> Any:
    global _R2_CLIENT
    if _R2_CLIENT is None:
        account = os.environ["R2_ACCOUNT_ID"]
        _R2_CLIENT = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )
    return _R2_CLIENT


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

Mode = Literal["i2v", "a2v", "distilled"]
_PIPELINES: dict[str, Any] = {}


def _distilled_lora_spec(strength: float) -> list[LoraPathStrengthAndSDOps]:
    return [
        LoraPathStrengthAndSDOps(
            str(DISTILLED_LORA),
            strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]


def get_pipeline(mode: Mode) -> Any:
    cached = _PIPELINES.get(mode)
    if cached is not None:
        return cached

    quantization = QuantizationPolicy.fp8_cast()
    common = {
        "spatial_upsampler_path": str(SPATIAL_UPSCALER),
        "gemma_root": str(GEMMA_ROOT),
        "loras": [],
        "quantization": quantization,
    }

    if mode == "i2v":
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=str(CHECKPOINT_22B),
            distilled_lora=_distilled_lora_spec(DISTILLED_LORA_STRENGTH),
            **common,
        )
    elif mode == "a2v":
        pipeline = A2VidPipelineTwoStage(
            checkpoint_path=str(CHECKPOINT_22B),
            distilled_lora=_distilled_lora_spec(DISTILLED_LORA_STRENGTH),
            **common,
        )
    elif mode == "distilled":
        # DistilledPipeline takes the distilled checkpoint directly and has no
        # separate distilled_lora argument.
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=str(CHECKPOINT_DISTILLED),
            **common,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    _PIPELINES[mode] = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class Input(BaseModel):
    mode: Mode
    prompt: str = Field(min_length=1)
    negative_prompt: str = NEGATIVE_PROMPT_DEFAULT
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    width: int = 768
    height: int = 1344
    num_frames: int = 121
    frame_rate: float = 25.0
    num_inference_steps: int = 40
    seed: Optional[int] = None
    image_strength: float = 1.0
    audio_start_time: float = 0.0
    audio_max_duration: Optional[float] = None
    enhance_prompt: bool = True

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0 or v <= 0:
            raise ValueError(f"must be a positive multiple of 32, got {v}")
        return v

    @field_validator("num_frames")
    @classmethod
    def _frame_count(cls, v: int) -> int:
        # LTX-2 requires num_frames of the form (8k + 1).
        if v < 9 or (v - 1) % 8 != 0:
            raise ValueError(f"num_frames must satisfy (8k+1) with k>=1, got {v}")
        return v

    @model_validator(mode="after")
    def _mode_requirements(self) -> "Input":
        if self.mode in ("i2v", "a2v", "distilled") and not self.image_url:
            raise ValueError(f"mode={self.mode} requires image_url")
        if self.mode == "a2v" and not self.audio_url:
            raise ValueError("mode=a2v requires audio_url")
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    started: float

    def done(self) -> float:
        return time.monotonic() - self.started


def _download(url: str, suffix: str, work: Path) -> Path:
    dst = work / f"{uuid.uuid4().hex}{suffix}"
    with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dst


def _suffix_from_url(url: str, default: str) -> str:
    path = url.split("?", 1)[0]
    if "." in path.rsplit("/", 1)[-1]:
        return "." + path.rsplit(".", 1)[-1].lower()
    return default


def _write_mp4(
    frames: Iterator[torch.Tensor],
    audio: Any,
    width: int,
    height: int,
    frame_rate: float,
    out_path: Path,
) -> None:
    """Pipe raw RGB frames into ffmpeg; mux with audio when present.

    The pipelines yield per-frame tensors. We materialise one frame at a time
    and write to ffmpeg stdin as raw RGB24 bytes, then optionally mux the
    returned Audio object (which exposes a .wav file path or raw samples).
    """
    video_only = out_path.with_suffix(".video.mp4")

    video_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{frame_rate}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart",
        str(video_only),
    ]
    proc = subprocess.Popen(video_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            # Expected shape: (H, W, 3) or (3, H, W), float [0,1] or uint8.
            t = frame
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.shape[0] == 3 and t.shape[-1] != 3:
                t = t.permute(1, 2, 0)
            if t.dtype != torch.uint8:
                t = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            proc.stdin.write(t.contiguous().cpu().numpy().tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg video encode failed")

    audio_path = _extract_audio_wav(audio)
    if audio_path is None:
        shutil.move(str(video_only), str(out_path))
        return

    mux_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_only),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    mux = subprocess.run(mux_cmd, check=False)
    if mux.returncode != 0:
        raise RuntimeError("ffmpeg mux failed")
    video_only.unlink(missing_ok=True)


def _extract_audio_wav(audio: Any) -> Optional[Path]:
    """Best-effort adapter for the Audio object LTX-2 returns.

    UPSTREAM-CONFLICT: the LTX-2 Audio class is not publicly documented at the
    revision this handler was written against. We try, in order:
      (1) `audio.path` attribute pointing at a wav/flac on disk
      (2) `audio.save(path)` or `audio.to_file(path)` method
      (3) `audio.waveform` tensor + `audio.sample_rate` attribute
      (4) None (i.e. pipeline returned silent output)
    If none of these work at runtime, this function raises so the failure is
    surfaced rather than silently producing a video-only MP4.
    """
    if audio is None:
        return None

    p = getattr(audio, "path", None)
    if p:
        return Path(p)

    for meth in ("save", "to_file", "write"):
        fn = getattr(audio, meth, None)
        if callable(fn):
            dst = Path(tempfile.mkstemp(suffix=".wav")[1])
            fn(str(dst))
            return dst if dst.stat().st_size > 0 else None

    waveform = getattr(audio, "waveform", None)
    sr = getattr(audio, "sample_rate", None)
    if waveform is not None and sr is not None:
        import numpy as np  # local import
        import wave

        arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.clip(arr, -1.0, 1.0)
        arr_i16 = (arr * 32767.0).astype(np.int16)
        dst = Path(tempfile.mkstemp(suffix=".wav")[1])
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(arr_i16.shape[0])
            w.setsampwidth(2)
            w.setframerate(int(sr))
            # wave expects interleaved frames.
            w.writeframes(arr_i16.T.tobytes() if arr_i16.shape[0] > 1 else arr_i16.tobytes())
        return dst if dst.stat().st_size > 0 else None

    # Length 0 / empty audio is legitimate for i2v / distilled.
    if getattr(audio, "__len__", None) and len(audio) == 0:
        return None

    raise RuntimeError(
        "LTX-2 Audio object did not expose .path / .save / .waveform — "
        "update _extract_audio_wav to match the installed revision."
    )


def _video_guider() -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[29],
    )


def _audio_guider() -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[29],
    )


def _run_pipeline(inp: Input, work: Path) -> tuple[Iterator[torch.Tensor], Any]:
    image_path: Optional[Path] = None
    if inp.image_url:
        image_path = _download(inp.image_url, _suffix_from_url(inp.image_url, ".png"), work)

    seed = inp.seed if inp.seed is not None else random.randint(0, 2**31 - 1)

    if inp.mode == "i2v":
        assert image_path is not None
        pipe = get_pipeline("i2v")
        images = [
            ImageConditioningInput(
                image_path=str(image_path),
                start_frame=0,
                strength=inp.image_strength,
                crf=40,
            )
        ]
        return pipe(
            prompt=inp.prompt,
            negative_prompt=inp.negative_prompt,
            seed=seed,
            height=inp.height,
            width=inp.width,
            num_frames=inp.num_frames,
            frame_rate=inp.frame_rate,
            num_inference_steps=inp.num_inference_steps,
            video_guider_params=_video_guider(),
            audio_guider_params=_audio_guider(),
            images=images,
            enhance_prompt=inp.enhance_prompt,
        )

    if inp.mode == "a2v":
        assert image_path is not None and inp.audio_url is not None
        audio_path = _download(inp.audio_url, _suffix_from_url(inp.audio_url, ".wav"), work)
        pipe = get_pipeline("a2v")
        # UPSTREAM-CONFLICT: A2VidPipelineTwoStage takes `images` as
        # list[tuple[str,int,float]] — bare tuples, not ImageConditioningInput.
        images = [(str(image_path), 0, inp.image_strength)]
        return pipe(
            prompt=inp.prompt,
            negative_prompt=inp.negative_prompt,
            seed=seed,
            height=inp.height,
            width=inp.width,
            num_frames=inp.num_frames,
            frame_rate=inp.frame_rate,
            num_inference_steps=inp.num_inference_steps,
            video_guider_params=_video_guider(),
            images=images,
            audio_path=str(audio_path),
            audio_start_time=inp.audio_start_time,
            audio_max_duration=inp.audio_max_duration,
            enhance_prompt=inp.enhance_prompt,
        )

    # distilled
    assert image_path is not None
    pipe = get_pipeline("distilled")
    images = [
        ImageConditioningInput(
            image_path=str(image_path),
            start_frame=0,
            strength=inp.image_strength,
            crf=40,
        )
    ]
    return pipe(
        prompt=inp.prompt,
        seed=seed,
        height=inp.height,
        width=inp.width,
        num_frames=inp.num_frames,
        frame_rate=inp.frame_rate,
        images=images,
        enhance_prompt=inp.enhance_prompt,
    )


def _upload_r2(local: Path, key: str, attempts: int = 3) -> str:
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            _r2().upload_file(str(local), os.environ["R2_BUCKET"], key)
            url = _r2().generate_presigned_url(
                "get_object",
                Params={"Bucket": os.environ["R2_BUCKET"], "Key": key},
                ExpiresIn=SIGNED_URL_TTL,
            )
            base = os.environ.get("R2_PUBLIC_BASE", "").rstrip("/")
            if base:
                return f"{base}/{key}"
            return url
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"R2 upload failed after {attempts} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Handler entrypoint
# ---------------------------------------------------------------------------


def _error(error_type: str, message: str, stage: Optional[str] = None) -> dict[str, Any]:
    return {
        "error_type": error_type,
        "error_message": message,
        "stage_failed": stage,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_id = event.get("id") or uuid.uuid4().hex
    payload = event.get("input") or {}

    try:
        inp = Input(**payload)
    except ValidationError as ve:
        return _error("validation", ve.json(), None)
    except Exception as e:  # noqa: BLE001
        return _error("validation", str(e), None)

    work = Path(tempfile.mkdtemp(prefix=f"ltx-{job_id}-"))
    out_mp4 = work / "out.mp4"
    timings: dict[str, float] = {}

    try:
        s_pipeline = Stage("pipeline", time.monotonic())
        frames, audio = _run_pipeline(inp, work)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return _error("pipeline", f"{type(e).__name__}: {e}", "stage1")

    try:
        s_encode = Stage("encode", time.monotonic())
        _write_mp4(frames, audio, inp.width, inp.height, inp.frame_rate, out_mp4)
        timings["pipeline_s"] = round(s_pipeline.done() - (time.monotonic() - s_encode.started), 3)
        timings["encode_s"] = round(s_encode.done(), 3)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return _error("pipeline", f"{type(e).__name__}: {e}", "encode")

    try:
        s_up = Stage("upload", time.monotonic())
        key = f"ltx-outputs/{job_id}/{int(time.time())}.mp4"
        url = _upload_r2(out_mp4, key)
        timings["upload_s"] = round(s_up.done(), 3)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return _error("upload", str(e), "upload")

    size_bytes = out_mp4.stat().st_size
    shutil.rmtree(work, ignore_errors=True)

    return {
        "url": url,
        "mode": inp.mode,
        "width": inp.width,
        "height": inp.height,
        "num_frames": inp.num_frames,
        "frame_rate": inp.frame_rate,
        "duration_s": round(inp.num_frames / inp.frame_rate, 3),
        "file_size_bytes": size_bytes,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Cold-start warmup
# ---------------------------------------------------------------------------

if DEFAULT_PIPELINE in ("i2v", "a2v", "distilled"):
    try:
        get_pipeline(DEFAULT_PIPELINE)  # type: ignore[arg-type]
    except Exception:
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
