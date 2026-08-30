# CUDA *devel*, not runtime, on purpose: torch routes some backward ops through
# Triton, which JIT-compiles a small helper that #includes cuda.h. The runtime
# image does not ship that header and training dies on the first backward pass.
#
# CUDA 13.0 to match the cu130 wheels plain pip resolves; Ubuntu 24.04 because
# it ships Python 3.12, which is the interpreter the venv+pip path was actually
# verified on.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# gcc stays in the final image because Triton shells out to it at runtime to
# build that helper; stripping build tools here breaks training, not the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks its system Python as externally managed, so pip refuses to
# install into it. A venv is the sanctioned way round that.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first so a code change does not re-download PyTorch.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Fails the build if the code cannot even be imported. Runs without a GPU: the
# one CUDA-requiring test skips itself.
RUN pytest -q tests

# Data is bind-mounted, never baked in: TITAN is access-gated and must not end
# up inside a redistributable image.
VOLUME ["/app/data", "/app/checkpoints"]

# Prints what the container can actually see. Override the command to train:
#
#   docker build -t titan-repro .
#   docker run --gpus all --rm titan-repro
#   docker run --gpus all --rm \
#       -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
#       titan-repro python -m titan.cli train --data-root data/titan --priors EP+IP+AP
#
# --gpus needs the NVIDIA Container Toolkit on the host.
CMD ["python", "scripts/check_gpu.py"]
