#!/bin/bash
# Build TXM pipeline Apptainer image.
#
# Usage:
#   bash build.sh                    # Build + test
#   bash build.sh --no-test          # Build only
#   bash build.sh --output /path/to  # Custom output dir
#
# The build takes ~30-60 min (PyTorch download + tomocupy compile).
# Consider running in a tmux/screen session.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/txm-pipeline.def"
OUTPUT_DIR="${SCRIPT_DIR}"
SIF_NAME="txm-pipeline.sif"

DO_TEST=true

for arg in "$@"; do
    case "$arg" in
        --no-test)      DO_TEST=false ;;
        --output)       shift; OUTPUT_DIR="$1" ;;
        --output=*)     OUTPUT_DIR="${arg#*=}" ;;
        *)              echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

SIF_PATH="${OUTPUT_DIR}/${SIF_NAME}"

# Use scratch for build temp if available (default /tmp may be too small)
if [ -n "${APPTAINER_TMPDIR:-}" ]; then
    mkdir -p "${APPTAINER_TMPDIR}"
fi

echo "=================================================="
echo " TXM Pipeline -- Apptainer Build"
echo " Def file:  ${DEF_FILE}"
echo " Output:    ${SIF_PATH}"
echo " Date:      $(date)"
echo "=================================================="

mkdir -p "${OUTPUT_DIR}"

# Back up previous .sif if it exists
if [ -f "${SIF_PATH}" ]; then
    BACKUP="${SIF_PATH}.bak.$(date +%Y%m%d-%H%M%S)"
    echo "Backing up existing SIF -> ${BACKUP}"
    mv "${SIF_PATH}" "${BACKUP}"
fi

echo ""
echo "[1/2] Building SIF from ${DEF_FILE}..."
echo ""

apptainer build --fakeroot "${SIF_PATH}" "${DEF_FILE}"

echo ""
echo "Build complete."
ls -lh "${SIF_PATH}"

# Test
if [ "${DO_TEST}" = true ]; then
    echo ""
    echo "[2/2] Running import tests..."
    apptainer exec "${SIF_PATH}" python -c "
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
    echo "[2/2] Skipping tests (--no-test)"
fi

echo ""
echo "=================================================="
echo " SIF:    ${SIF_PATH}"
echo " Size:   $(du -h "${SIF_PATH}" | cut -f1)"
echo "=================================================="
