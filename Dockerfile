# RunPod serverless image for LTX-2.3 video generation.
#
# Base image: the RunPod-maintained PyTorch 2.7 / CUDA 12.8 devel image.
# If a newer tag is published, bump LTX_BASE_IMAGE and the expected CUDA
# version documented in README.md. Verify the tag exists on Docker Hub
# (https://hub.docker.com/r/runpod/pytorch/tags) before building.
ARG LTX_BASE_IMAGE=runpod/pytorch:2.7.0-py3.12-cuda12.8.0-cudnn-devel-ubuntu22.04
FROM ${LTX_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/ltx-2/.venv/bin:/root/.local/bin:${PATH}

# System deps: ffmpeg for MP4 muxing, git for cloning LTX-2, build-essential
# for any C extensions that may be compiled by uv sync.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ca-certificates curl ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install uv (the package manager LTX-2 uses for its lockfile).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone LTX-2 at a pinned commit.
# BUMP STEPS:
#   1) run `git ls-remote https://github.com/Lightricks/LTX-2 HEAD`
#   2) paste the 40-char SHA into LTX_COMMIT_SHA below
#   3) rebuild the image
ARG LTX_COMMIT_SHA=41d924371612b692c0fd1e4d9d94c3dfb3c02cb3
RUN git clone https://github.com/Lightricks/LTX-2.git /opt/ltx-2 \
 && cd /opt/ltx-2 \
 && git checkout "${LTX_COMMIT_SHA}"

WORKDIR /opt/ltx-2

# Install LTX-2 python deps + xformers extra (Flash-Attention wheels).
# Note: Flash-Attention 3 only accelerates on Hopper (H100/H200). On other GPUs
# the installed wheels still work but use the attention fallback.
RUN uv sync --frozen --extra xformers

# Serverless-specific deps (not pulled by LTX-2 itself).
RUN /opt/ltx-2/.venv/bin/pip install --no-cache-dir \
      runpod==1.7.* \
      boto3==1.34.* \
      httpx==0.27.* \
      pydantic==2.*

# Ship the handler.
COPY handler.py /opt/ltx-2/handler.py

WORKDIR /opt/ltx-2
CMD ["/opt/ltx-2/.venv/bin/python", "-u", "handler.py"]
