#!/bin/bash
# Container entrypoint for TXM pipeline.
#
# Sets up NVIDIA library paths and forwards arguments.
set -euo pipefail

# Add pip-installed nvidia libs to LD_LIBRARY_PATH (cupy/torch ship their own)
NVIDIA_LIBS="/usr/local/lib/python3.11/dist-packages/nvidia"
if [ -d "${NVIDIA_LIBS}" ]; then
    for subdir in cuda_nvrtc cuda_runtime cudnn cublas cufft curand cusolver cusparse; do
        if [ -d "${NVIDIA_LIBS}/${subdir}/lib" ]; then
            export LD_LIBRARY_PATH="${NVIDIA_LIBS}/${subdir}/lib:${LD_LIBRARY_PATH:-}"
        fi
    done
fi

# Forward all arguments as the command to run
exec "$@"
