# Ultra compute service — GPU node image.
#
# One image that IS the GPU runtime: the FastAPI framework runs yolov5 train/val +
# rarespot detect as fresh subprocesses / in-process, so there is no ssh, no rsync,
# no nested docker, and no per-run pip install. Everything the executors need is
# baked at build time (that is what makes back-to-back launches deterministic).
#
# Build context = repo root:
#   docker build -f services/compute_service/Dockerfile -t ultra-compute:<sha> .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet \
    YOLOv5_AUTOINSTALL=false \
    ULTRA_RARESPOT_YOLOV5_PATH=/opt/rarespot/yolov5 \
    PYTHONPATH=/app:/opt/ultra

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/compute_service/requirements.txt /tmp/compute-requirements.txt
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -r /tmp/compute-requirements.txt

# The rarespot/training compute code, installed WITHOUT its agent deps (langchain,
# nats, ...). Its import chain is torch/PIL/numpy only — verified in the build test.
COPY backend/deepagents_runtime/src/ultra_deepagents /opt/ultra/ultra_deepagents

# The BN-patched yolov5 fork, baked once at /opt/rarespot/yolov5.
COPY third_party/yolov5 /opt/rarespot/yolov5
RUN python3.10 -c "from ultra_deepagents.training.rarespot_adapter import _apply_bn_freeze_patch; from pathlib import Path; _apply_bn_freeze_patch(Path('/opt/rarespot/yolov5/train.py')); print('BN-freeze patch applied')"

# Seed Arial.ttf from matplotlib's DejaVuSans so yolov5's check_font never attempts
# its offline-404 download (kept out of the eval path too via _seed_yolov5_font).
RUN mkdir -p /root/.config/Ultralytics && python3.10 -c "import matplotlib, pathlib, shutil; shutil.copy(pathlib.Path(matplotlib.__file__).parent/'mpl-data'/'fonts'/'ttf'/'DejaVuSans.ttf', '/root/.config/Ultralytics/Arial.ttf'); print('seeded Arial.ttf')"

COPY services /app/services

# Fail the build if the framework import is broken (executors load defensively at
# runtime, but the app + registry must import cleanly)...
RUN python3.10 -c "import services.compute_service.app as a; print('compute app import OK')"
# ...and if the rarespot executors' full dependency chain (torch + ultra_deepagents
# + the baked yolov5/hyp) does not import + register cleanly. Fail fast at build.
RUN python3.10 -c "import services.compute_service.executors as e; e.load_builtin_executors(only={'rarespot_train','rarespot_evaluate','rarespot_smoke'}); errs=e.load_errors(); keys=sorted(m.key for m in e.list_manifests()); print('executors:', keys, 'errors:', errs); assert {'rarespot.train','rarespot.evaluate','rarespot.smoke'} <= set(keys), errs"

EXPOSE 8030

CMD ["python3.10", "-m", "uvicorn", "services.compute_service.app:app", "--host", "0.0.0.0", "--port", "8030", "--workers", "1"]
