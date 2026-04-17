FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/megaseg_service/requirements.txt /tmp/megaseg-requirements.txt
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -r /tmp/megaseg-requirements.txt && \
    python3.10 -m pip install --no-deps \
      bioio==1.1.0 \
      bioio-ome-zarr==1.2.0 \
      bioio-ome-tiff==1.0.0 \
      bioio-tifffile==1.0.0

COPY src /app/src
COPY services /app/services

EXPOSE 8010

CMD ["python3.10", "-m", "uvicorn", "services.megaseg_service.app:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
