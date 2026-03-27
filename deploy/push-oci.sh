#!/bin/bash
# Push the Apptainer SIF to an OCI container registry.
#
# Converts the Apptainer SIF to an OCI image and pushes to a registry,
# making it pullable by Docker, Kubernetes, AWS ECS/EKS, etc.
#
# Usage:
#   bash push-oci.sh txm-pipeline.sif ghcr.io/osmiumzero/tomocloud-pipeline:latest
#   bash push-oci.sh txm-pipeline.sif <registry>/<org>/<image>:<tag>
#
# Prerequisites:
#   - apptainer >= 1.1 (for OCI push support)
#   - Registry credentials configured (e.g., docker login, podman login)
#
# For GHCR (GitHub Container Registry):
#   echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <sif-file> <registry/org/image:tag>"
    echo ""
    echo "Examples:"
    echo "  $0 txm-pipeline.sif ghcr.io/osmiumzero/tomocloud-pipeline:latest"
    echo "  $0 txm-pipeline.sif docker.io/myuser/tomocloud-pipeline:v1.0"
    exit 1
fi

SIF_FILE="$1"
REGISTRY_TARGET="$2"

if [ ! -f "${SIF_FILE}" ]; then
    echo "ERROR: SIF file not found: ${SIF_FILE}"
    exit 1
fi

echo "=================================================="
echo " Push SIF to OCI Registry"
echo " SIF:    ${SIF_FILE}"
echo " Target: ${REGISTRY_TARGET}"
echo " Date:   $(date)"
echo "=================================================="

echo ""
echo "Pushing SIF to OCI registry..."
apptainer push "${SIF_FILE}" "oras://${REGISTRY_TARGET}"

echo ""
echo "Push complete."
echo ""
echo "Pull with:"
echo "  apptainer pull ${REGISTRY_TARGET##*/}.sif oras://${REGISTRY_TARGET}"
echo "  docker pull ${REGISTRY_TARGET}"
echo "=================================================="
