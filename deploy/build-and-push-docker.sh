#!/bin/bash
# Build the Docker/OCI image and push to a container registry.
#
# Usage:
#   bash build-and-push-docker.sh ghcr.io/osmiumzero/tomocloud-pipeline:latest
#   bash build-and-push-docker.sh <registry>/<org>/<image>:<tag>
#
# Prerequisites:
#   - Docker (or podman) installed
#   - NVIDIA Container Toolkit (for GPU test)
#   - Registry credentials configured
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <registry/org/image:tag> [--no-test] [--no-push]"
    exit 1
fi

REGISTRY_TARGET="$1"
shift

DO_TEST=true
DO_PUSH=true

for arg in "$@"; do
    case "$arg" in
        --no-test) DO_TEST=false ;;
        --no-push) DO_PUSH=false ;;
        *)         echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "${SCRIPT_DIR}")/docker"

echo "=================================================="
echo " TXM Pipeline -- Docker Build + Push"
echo " Dockerfile: ${DOCKER_DIR}/Dockerfile"
echo " Target:     ${REGISTRY_TARGET}"
echo " Date:       $(date)"
echo "=================================================="

echo ""
echo "[1/3] Building Docker image..."
docker build -t "${REGISTRY_TARGET}" -f "${DOCKER_DIR}/Dockerfile" "${DOCKER_DIR}"

echo ""
echo "Build complete."

if [ "${DO_TEST}" = true ]; then
    echo ""
    echo "[2/3] Running import tests..."
    docker run --rm "${REGISTRY_TARGET}" python -c "
import numpy; print(f'numpy {numpy.__version__}')
import scipy; print(f'scipy {scipy.__version__}')
import cupy; print(f'cupy {cupy.__version__}')
import tomocupy; print('tomocupy OK')
import torch; print(f'torch {torch.__version__}')
import sam2; print('sam2 OK')
import zarr; print(f'zarr {zarr.__version__}')
import globus_sdk; print('globus_sdk OK')
print('All imports OK')
"
    echo "Import tests passed."
else
    echo "[2/3] Skipping tests (--no-test)"
fi

if [ "${DO_PUSH}" = true ]; then
    echo ""
    echo "[3/3] Pushing to registry..."
    docker push "${REGISTRY_TARGET}"
    echo ""
    echo "Push complete."
    echo "Pull with: docker pull ${REGISTRY_TARGET}"
else
    echo "[3/3] Skipping push (--no-push)"
fi

echo "=================================================="
