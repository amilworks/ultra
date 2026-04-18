FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        opencv-python-headless \
        scikit-image \
        imageio \
        tifffile \
        Pillow \
        statsmodels \
        joblib \
        networkx \
        numba

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "--version"]
