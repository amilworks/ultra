#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODULE_DIR="${ROOT_DIR}/source/modules/SimpleUniversalProcess"
IMAGE_TAG="${IMAGE_TAG:-simpleuniversalprocess:v1.0.0}"
REGISTRY_TAG="${REGISTRY_TAG:-nail04.ece.ucsb.edu:5000/simpleuniversalprocess:v1.0.0}"
DOCKER_BUILD_PLATFORM="${DOCKER_BUILD_PLATFORM:-linux/amd64}"

docker build --platform "${DOCKER_BUILD_PLATFORM}" -t "${IMAGE_TAG}" "${MODULE_DIR}"
docker tag "${IMAGE_TAG}" "${REGISTRY_TAG}"

echo "built_image=${IMAGE_TAG}"
echo "tagged_image=${REGISTRY_TAG}"
