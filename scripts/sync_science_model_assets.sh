#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST=""
REMOTE_ROOT="/srv/ultra"

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_science_model_assets.sh --source /path/to/old/repo [--remote-host user@host] [--remote-root /srv/ultra]

This syncs the minimum local science-model assets needed for production-safe YOLO/SAM usage:
  - YOLO checkpoints: RareSpotWeights.pt, yolo26x.pt, yolo26n.pt
  - SAM3 local snapshot: data/models/sam3/facebook-sam3/
  - MedSAM2 checkpoints: data/models/medsam2/checkpoints/
  - MedSAM2 runtime: third_party/MedSAM2/

Local targets inside this repo:
  data/models/yolo/
  data/models/sam3/facebook-sam3/
  data/models/medsam2/checkpoints/
  data/runtime/MedSAM2/

If --remote-host is provided, the same assets are also synced to:
  <remote-root>/models/yolo/
  <remote-root>/models/sam3/facebook-sam3/
  <remote-root>/models/medsam2/checkpoints/
  <remote-root>/runtime/MedSAM2/
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE_REPO="${2:-}"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="${2:-}"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$SOURCE_REPO" ]; then
  echo "--source is required" >&2
  usage >&2
  exit 1
fi

SOURCE_REPO="$(cd "$SOURCE_REPO" && pwd)"

require_path() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "Missing required asset: $path" >&2
    exit 1
  fi
}

SOURCE_RARESPOT="$SOURCE_REPO/RareSpotWeights.pt"
SOURCE_YOLO26X="$SOURCE_REPO/yolo26x.pt"
SOURCE_YOLO26N="$SOURCE_REPO/yolo26n.pt"
SOURCE_SAM3_DIR="$SOURCE_REPO/data/models/sam3/facebook-sam3"
SOURCE_MEDSAM2_CHECKPOINTS="$SOURCE_REPO/data/models/medsam2/checkpoints"
SOURCE_MEDSAM2_RUNTIME="$SOURCE_REPO/third_party/MedSAM2"

require_path "$SOURCE_RARESPOT"
require_path "$SOURCE_YOLO26X"
require_path "$SOURCE_YOLO26N"
require_path "$SOURCE_SAM3_DIR"
require_path "$SOURCE_MEDSAM2_CHECKPOINTS"
require_path "$SOURCE_MEDSAM2_RUNTIME"

LOCAL_YOLO_DIR="$REPO_ROOT/data/models/yolo"
LOCAL_SAM3_DIR="$REPO_ROOT/data/models/sam3/facebook-sam3"
LOCAL_MEDSAM2_CHECKPOINT_DIR="$REPO_ROOT/data/models/medsam2/checkpoints"
LOCAL_MEDSAM2_RUNTIME_DIR="$REPO_ROOT/data/runtime/MedSAM2"

mkdir -p \
  "$LOCAL_YOLO_DIR" \
  "$LOCAL_SAM3_DIR" \
  "$LOCAL_MEDSAM2_CHECKPOINT_DIR" \
  "$LOCAL_MEDSAM2_RUNTIME_DIR"

rsync -a "$SOURCE_RARESPOT" "$LOCAL_YOLO_DIR/RareSpotWeights.pt"
rsync -a "$SOURCE_YOLO26X" "$LOCAL_YOLO_DIR/yolo26x.pt"
rsync -a "$SOURCE_YOLO26N" "$LOCAL_YOLO_DIR/yolo26n.pt"
rsync -a --delete "$SOURCE_SAM3_DIR/" "$LOCAL_SAM3_DIR/"
rsync -a "$SOURCE_MEDSAM2_CHECKPOINTS/" "$LOCAL_MEDSAM2_CHECKPOINT_DIR/"
rsync -a --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.egg-info' \
  --exclude 'build' \
  --exclude 'data' \
  --exclude 'notebooks' \
  --exclude 'checkpoints' \
  "$SOURCE_MEDSAM2_RUNTIME/" "$LOCAL_MEDSAM2_RUNTIME_DIR/"

if [ -n "$REMOTE_HOST" ]; then
  ssh "$REMOTE_HOST" "mkdir -p \
    '$REMOTE_ROOT/models/yolo' \
    '$REMOTE_ROOT/models/sam3/facebook-sam3' \
    '$REMOTE_ROOT/models/medsam2/checkpoints' \
    '$REMOTE_ROOT/runtime/MedSAM2'"
  rsync -a "$SOURCE_RARESPOT" "$REMOTE_HOST:$REMOTE_ROOT/models/yolo/RareSpotWeights.pt"
  rsync -a "$SOURCE_YOLO26X" "$REMOTE_HOST:$REMOTE_ROOT/models/yolo/yolo26x.pt"
  rsync -a "$SOURCE_YOLO26N" "$REMOTE_HOST:$REMOTE_ROOT/models/yolo/yolo26n.pt"
  rsync -a --delete "$SOURCE_SAM3_DIR/" "$REMOTE_HOST:$REMOTE_ROOT/models/sam3/facebook-sam3/"
  rsync -a "$SOURCE_MEDSAM2_CHECKPOINTS/" "$REMOTE_HOST:$REMOTE_ROOT/models/medsam2/checkpoints/"
  rsync -a --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '*.egg-info' \
    --exclude 'build' \
    --exclude 'data' \
    --exclude 'notebooks' \
    --exclude 'checkpoints' \
    "$SOURCE_MEDSAM2_RUNTIME/" "$REMOTE_HOST:$REMOTE_ROOT/runtime/MedSAM2/"
fi

echo "Synced science-model assets from $SOURCE_REPO"
echo "Local repo targets:"
echo "  $LOCAL_YOLO_DIR"
echo "  $LOCAL_SAM3_DIR"
echo "  $LOCAL_MEDSAM2_CHECKPOINT_DIR"
echo "  $LOCAL_MEDSAM2_RUNTIME_DIR"
if [ -n "$REMOTE_HOST" ]; then
  echo "Remote host targets:"
  echo "  $REMOTE_HOST:$REMOTE_ROOT/models/yolo"
  echo "  $REMOTE_HOST:$REMOTE_ROOT/models/sam3/facebook-sam3"
  echo "  $REMOTE_HOST:$REMOTE_ROOT/models/medsam2/checkpoints"
  echo "  $REMOTE_HOST:$REMOTE_ROOT/runtime/MedSAM2"
fi
