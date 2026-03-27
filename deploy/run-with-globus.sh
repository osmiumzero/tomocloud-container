#!/bin/bash
# Run the pipeline container with Globus credentials and HuggingFace cache.
#
# This script handles the common bind mounts needed for:
#   - Globus SDK tokens (for data transfer)
#   - HuggingFace cache (for SAM3 gated model access)
#   - Input/output data directories
#
# Usage (Apptainer):
#   bash run-with-globus.sh apptainer \
#     --sif /path/to/txm-pipeline.sif \
#     --data-dir /path/to/data \
#     --output-dir /path/to/output \
#     -- python /app/reconstruct_gpu.py ...
#
# Usage (Docker):
#   bash run-with-globus.sh docker \
#     --image ghcr.io/osmiumzero/tomocloud-pipeline:latest \
#     --data-dir /path/to/data \
#     --output-dir /path/to/output \
#     -- python /app/reconstruct_gpu.py ...
set -euo pipefail

RUNTIME="${1:-}"
shift || true

SIF_PATH=""
IMAGE=""
DATA_DIR=""
OUTPUT_DIR=""
MODEL_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sif)        SIF_PATH="$2"; shift 2 ;;
        --image)      IMAGE="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --)           shift; EXTRA_ARGS=("$@"); break ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

if [ "${RUNTIME}" = "apptainer" ]; then
    if [ -z "${SIF_PATH}" ]; then
        echo "ERROR: --sif required for apptainer runtime"
        exit 1
    fi

    BIND_ARGS=""
    [ -n "${DATA_DIR}" ]   && BIND_ARGS="${BIND_ARGS} --bind ${DATA_DIR}:/data"
    [ -n "${OUTPUT_DIR}" ] && BIND_ARGS="${BIND_ARGS} --bind ${OUTPUT_DIR}:/output"
    [ -n "${MODEL_DIR}" ]  && BIND_ARGS="${BIND_ARGS} --bind ${MODEL_DIR}:/models"
    [ -d "${HF_CACHE}" ]   && BIND_ARGS="${BIND_ARGS} --bind ${HF_CACHE}:/hf_cache"

    apptainer exec --nv ${BIND_ARGS} \
        --env HF_HOME=/hf_cache \
        "${SIF_PATH}" \
        "${EXTRA_ARGS[@]}"

elif [ "${RUNTIME}" = "docker" ]; then
    if [ -z "${IMAGE}" ]; then
        echo "ERROR: --image required for docker runtime"
        exit 1
    fi

    MOUNT_ARGS=""
    [ -n "${DATA_DIR}" ]   && MOUNT_ARGS="${MOUNT_ARGS} -v ${DATA_DIR}:/data"
    [ -n "${OUTPUT_DIR}" ] && MOUNT_ARGS="${MOUNT_ARGS} -v ${OUTPUT_DIR}:/output"
    [ -n "${MODEL_DIR}" ]  && MOUNT_ARGS="${MOUNT_ARGS} -v ${MODEL_DIR}:/models"
    [ -d "${HF_CACHE}" ]   && MOUNT_ARGS="${MOUNT_ARGS} -v ${HF_CACHE}:/hf_cache"

    docker run --gpus all --rm \
        ${MOUNT_ARGS} \
        -e HF_HOME=/hf_cache \
        "${IMAGE}" \
        "${EXTRA_ARGS[@]}"

else
    echo "Usage: $0 <apptainer|docker> [options] -- <command>"
    exit 1
fi
