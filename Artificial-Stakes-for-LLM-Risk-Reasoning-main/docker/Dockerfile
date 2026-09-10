# CUDA 12.4 runtime + Python 3.11 + vLLM for Llama-3.1-8B inference on an RTX 6000.
# Build: docker build -f docker/Dockerfile -t risk-reasoning:latest .
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.hf_cache \
    TRANSFORMERS_CACHE=/workspace/.hf_cache \
    TORCH_HOME=/workspace/.torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the package in editable mode with vllm + dev extras. We copy
# pyproject.toml first so dependency installs are cached independently of source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -e ".[vllm,dev]"

COPY . .

CMD ["bash"]
