# RunPod serverless image for LTX-2.3 video generation.
#
# Base image: RunPod-maintained PyTorch 2.7.1 / CUDA 12.9 / Ubuntu 22.04.
# Naming scheme is runpod/pytorch:<img>-cu<cuda>-torch<torch>-ubuntu<ver>.
# Verify new tags on https://hub.docker.com/r/runpod/pytorch/tags before bumping.
# The base ships Python 3.11; `uv sync` inside LTX-2 installs the 3.12 toolchain
# this project requires.
ARG LTX_BASE_IMAGE=runpod/pytorch:1.0.3-cu1290-torch271-ubuntu2204
FROM ${LTX_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    UV_LINK_MODE=copy \
    PATH=/root/.local/bin:${PATH}

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
# The RunPod base image ships with UV_SYSTEM_PYTHON=1, so `uv sync` installs
# directly into the image's system Python (no project venv). We keep that
# default and layer our serverless deps on top with plain pip.
# Note: Flash-Attention 3 only accelerates on Hopper (H100/H200). On other GPUs
# the installed wheels still work but use the attention fallback.
RUN uv sync --frozen --extra xformers

# Serverless-specific deps (not pulled by LTX-2 itself).
RUN python -m pip install --no-cache-dir \
      runpod==1.7.* \
      boto3==1.34.* \
      httpx==0.27.* \
      pydantic==2.*

# Ship the handler.
COPY handler.py /opt/ltx-2/handler.py

WORKDIR /opt/ltx-2
CMD ["python", "-u", "handler.py"]
