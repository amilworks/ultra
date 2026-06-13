#!/usr/bin/env bash
# Launch the libbioimage image-service container: the FastAPI decode/convert
# service (port 8099) plus the convert worker that turns uploads into tiled
# pyramids (image.derive_pyramid jobs). The Go control plane proxies its V2 image
# endpoints to this service (set ULTRA_CONTROL_IMAGE_SERVICE_URL=http://127.0.0.1:8099).
#
# Prereq: the engine build at $ULTRA_IMGSVC_BUILD_DIR (libimgcnv.so + the python
# binding wheel), produced by imaging-engine-build/apply-engine-patches.sh, and a
# docker image tagged $ULTRA_IMGSVC_IMAGE with imgcnv's runtime deps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${ULTRA_IMGSVC_NAME:-ultra-imgsvc}"
IMAGE="${ULTRA_IMGSVC_IMAGE:-imgcnv:build}"
BUILD_DIR="${ULTRA_IMGSVC_BUILD_DIR:-/tmp/bioimageconvert}"
PORT="${ULTRA_IMGSVC_PORT:-8099}"
WORKERS="${ULTRA_IMGSVC_WORKERS:-2}"
UPLOADS="${ULTRA_CONTROL_UPLOAD_ROOT:-$ROOT/data/uploads}"
NATS_URL="${ULTRA_IMGSVC_NATS_URL:-nats://host.docker.internal:4222}"
JOBS_SUBJECT="${ULTRA_IMGSVC_JOBS_SUBJECT:-ultra.data_agent.jobs}"
WORKER_DURABLE="${ULTRA_IMGSVC_WORKER_DURABLE:-ultra-image-convert-worker}"

if [ ! -d "$BUILD_DIR/bin" ]; then
  echo "engine build not found at $BUILD_DIR/bin (run imaging-engine-build/apply-engine-patches.sh)" >&2
  exit 1
fi

echo "[image-service] (re)launching $NAME on :$PORT (uploads=$UPLOADS, jobs=$JOBS_SUBJECT)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  -p "${PORT}:8099" \
  --add-host host.docker.internal:host-gateway \
  -v "$BUILD_DIR:/build" \
  -v "$ROOT/backend/deepagents_runtime:/app" \
  -v "$UPLOADS:$UPLOADS" \
  -e ULTRA_CONTROL_NATS_URL="$NATS_URL" \
  -e ULTRA_CONTROL_NATS_IMAGE_JOBS_SUBJECT="$JOBS_SUBJECT" \
  -e ULTRA_CONTROL_NATS_IMAGE_WORKER_DURABLE="$WORKER_DURABLE" \
  -e ULTRA_IMGSVC_WORKERS="$WORKERS" \
  "$IMAGE" bash -lc '
    set -e
    export LD_LIBRARY_PATH=/build/bin PYTHONPATH=/app/src PATH=/build/bin:$PATH
    cp /build/bin/libimgcnv.so.* /usr/local/lib/ 2>/dev/null || true; ldconfig 2>/dev/null || true
    # ffmpeg/ffprobe for video posters (libbioimage parses video containers but its
    # decoder is non-functional in this build). Idempotent across restarts.
    command -v ffmpeg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y --no-install-recommends ffmpeg; } >/dev/null 2>&1 || true
    pip install --quiet /build/python pillow fastapi uvicorn nats-py 2>&1 | tail -1
    # Convert worker: source -> tiled BigTIFF pyramid (image.derive_pyramid jobs).
    python3 -m ultra_deepagents.image_convert_worker >/tmp/convert-worker.log 2>&1 &
    # Image service: decode/convert over libbioimage (process pool).
    exec python3 -m ultra_deepagents.image_service --host 0.0.0.0 --port 8099 --workers "${ULTRA_IMGSVC_WORKERS:-2}"
  '

echo "[image-service] waiting for health..."
for i in $(seq 1 40); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    echo "[image-service] healthy on :$PORT"
    exit 0
  fi
  sleep 1
done
echo "[image-service] did not become healthy in time; check: docker logs $NAME" >&2
exit 1
