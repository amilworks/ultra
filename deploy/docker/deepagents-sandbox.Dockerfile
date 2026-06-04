FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        h5py \
        imageio \
        matplotlib \
        networkx \
        nibabel \
        numpy \
        opencv-python-headless \
        pandas \
        pillow \
        pyarrow \
        scikit-image \
        scikit-learn \
        scipy \
        seaborn \
        tifffile

WORKDIR /workspace
