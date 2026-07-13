# syntax=docker/dockerfile:1.7

# Kawin 0.5 requires NumPy 2. This image is intentionally separate from Ultra's
# shared NumPy-1.26 execution sandbox and is invoked only through the typed CLI.
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV HOME=/tmp \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TZ=UTC \
    XDG_CACHE_HOME=/tmp/cache

COPY deploy/docker/materials-kinetics-requirements.lock /opt/ultra/requirements.lock

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install \
      pip==25.1.1 \
      setuptools==83.0.0 \
      wheel==0.45.1 \
    && python -m pip install --require-hashes -r /opt/ultra/requirements.lock \
    && python -m pip check \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only the closed typed runtime. Do not copy the shared materials package
# or arbitrary execution machinery into this image.
COPY backend/deepagents_runtime/src/ultra_deepagents/kinetics_runtime \
     /usr/local/lib/python3.11/site-packages/ultra_deepagents/kinetics_runtime
RUN : > /usr/local/lib/python3.11/site-packages/ultra_deepagents/__init__.py

# The scientific qualification suite uses real Kawin test databases and executes
# transport, finite-volume diffusion, and KWN precipitation during the image build.
COPY backend/deepagents_runtime/tests/kinetics_runtime/test_kawin_runtime.py \
     /opt/ultra/tests/test_kawin_runtime.py
RUN python -I -m ultra_deepagents.kinetics_runtime.cli --self-check \
    && python -I -m pytest -q /opt/ultra/tests/test_kawin_runtime.py

RUN groupadd --gid 10001 ultra \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp ultra \
    && mkdir -p /workspace /outputs \
    && chown 10001:10001 /workspace /outputs

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Ultra isolated materials kinetics runtime" \
      org.opencontainers.image.description="Pinned Kawin NumPy-2 typed runtime; orchestrator must use --network none and immutable image ID" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/amilworks/ultra"

USER 10001:10001
WORKDIR /workspace
CMD ["python", "-I", "-m", "ultra_deepagents.kinetics_runtime.cli", "--self-check"]
