FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/codeexec_service/requirements.txt /tmp/codeexec-service-requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/codeexec-service-requirements.txt

COPY src /app/src
COPY services /app/services

EXPOSE 8020

CMD ["python", "-m", "uvicorn", "services.codeexec_service.app:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
