# TXM Pipeline Container -- OCI/Docker Image
#
# Multi-stage build:
#   Stage 1 (builder): CUDA 12.2 devel image -- builds tomocupy from source
#   Stage 2 (runtime): CUDA 12.2 runtime image -- final slim image
#
# Build:
#   docker build -t ghcr.io/<org>/tomocloud-pipeline:latest .
#
# Run:
#   docker run --gpus all -v /data:/data ghcr.io/<org>/tomocloud-pipeline:latest \
#     python /app/reconstruct_gpu.py --help
#
# CUDA 12.2 chosen for broad driver compatibility (535+).

# -- Stage 1: Builder (compile tomocupy) --
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    gcc g++ cmake swig git pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Build deps for tomocupy
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    cupy-cuda12x \
    scikit-build \
    cmake \
    scipy \
    numexpr \
    opencv-python-headless \
    pywavelets

# Clone and build tomocupy from source
# MUST use --no-build-isolation: pip's isolated build env produces .so
# files with no embedded CUDA device code
RUN git clone https://github.com/tomography/tomocupy.git /tmp/tomocupy \
    && cd /tmp/tomocupy \
    && pip install --no-build-isolation --no-cache-dir --no-deps .


# -- Stage 2: Runtime --
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 + minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Core numerical stack (numpy<2.0 is critical for tomocupy ABI)
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    "scipy>=1.13,<1.15" \
    cupy-cuda12x \
    h5py \
    hdf5plugin \
    tifffile \
    numexpr \
    opencv-python-headless \
    pywavelets

# Zarr / OME-Zarr
RUN pip install --no-cache-dir \
    zarr \
    ome-zarr \
    numcodecs

# PyTorch (cu121 wheels -- runtime CUDA 12.1 compatible with 12.2 driver)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# SAM runtime deps
RUN pip install --no-cache-dir \
    "setuptools<70" \
    hydra-core iopath tqdm ftfy regex timm einops \
    pycocotools psutil huggingface-hub

# SAM 2.1
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    git+https://github.com/facebookresearch/sam2.git

# SAM3
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    git+https://github.com/facebookresearch/sam3.git

# Globus SDK (for in-job data transfers)
RUN pip install --no-cache-dir "globus-sdk" "cryptography<43"

# Copy tomocupy built artifacts from builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages/tomocupy /usr/local/lib/python3.11/dist-packages/tomocupy
COPY --from=builder /usr/local/lib/python3.11/dist-packages/tomocupy*.dist-info /usr/local/lib/python3.11/dist-packages/

# Re-pin numpy last (pip may have upgraded it via transitive deps)
RUN pip install --no-cache-dir "numpy<2.0"

# Container entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment
ENV PYTHONPATH="/app:${PYTHONPATH:-}" \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    HF_HOME="/hf_cache"

WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
