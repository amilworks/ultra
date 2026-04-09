#!/usr/bin/env bash
set -euo pipefail

RELEASE_SHA="${1:-}"
if [ -z "$RELEASE_SHA" ]; then
  echo "Usage: $0 <git-sha>" >&2
  exit 1
fi

ULTRA_RELEASE_ROOT="${ULTRA_RELEASE_ROOT:-/srv/ultra}"
RELEASE_DIR="$ULTRA_RELEASE_ROOT/releases/$RELEASE_SHA/frontend"
CURRENT_LINK="$ULTRA_RELEASE_ROOT/frontend-current"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Frontend release directory not found: $RELEASE_DIR" >&2
  exit 1
fi

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
nginx -t
systemctl reload nginx

echo "Frontend deploy complete for $RELEASE_SHA"
