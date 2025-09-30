
# CUDA image suitable for PyTorch 2.3+ on CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# (Optional) prefetch weights at build time
# RUN python scripts/download_weights.py || true

EXPOSE 8093
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8093"]
