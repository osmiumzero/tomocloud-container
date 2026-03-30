#!/bin/bash
# Full pipeline for AWS/Kubernetes (IRI API).
# Adapted from S3DF run_full_pipeline.sh — no Slurm, no Apptainer.
#
# Stages:
#   1. normalize_raw.py    — GPU flat-field normalization
#   2. reconstruct_gpu.py  — GPU reconstruction (tomocupy)
#   3. tiff_to_zarr.py     — TIFF->OME-Zarr with pyramids
#   4. sam3_segment.py     — SAM3 segmentation (all GPUs via torchrun)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/home/timdunn/.cache/huggingface}"

NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

echo "=== TomoCloud Full Pipeline (AWS) ==="
echo "Node: $(hostname)"
echo "GPUs: ${NGPUS}"
echo "Python: $(which python)"
echo "Start time: $(date -Iseconds)"
echo "Arguments: $@"
echo "=============================="

# Parse arguments
DATA_DIR=""
OUTPUT_DIR=""
H5_FILE=""

# Recon args
ALGORITHM="fourierrec"
DTYPE="float32"
NSINO="8"
NPROJ="8"
FBP_FILTER="parzen"
ROTATION_AXIS="-1.0"
REMOVE_STRIPE_METHOD="none"

# Zarr args
ZARR_CHUNK_SIZE="128"
ZARR_COMPRESSION="blosc-lz4"
ZARR_NUM_LEVELS="4"
ZARR_PIXEL_SIZE_NM="0"
ZARR_NGPUS="4"
ZARR_IO_WORKERS="8"
ZARR_PREVIEW_SLICES="100"

# SAM3 args
SAM3_BACKEND="sam3"
SAM3_TEXT_PROMPT="circle"
SAM3_TEXT_PROMPTS=""
SAM3_CONFIDENCE="0.05"
SAM3_MODEL_CHECKPOINT="/data/home/timdunn/models/sam2.1_hiera_large.pt"
SAM3_MODEL_CONFIG="sam2.1_hiera_l"
SAM3_AXIS="z"
SAM3_BATCH_SIZE="4"
SAM3_POINTS_PER_SIDE="32"
SAM3_PRED_IOU_THRESH="0.7"
SAM3_STABILITY_SCORE_THRESH="0.8"
SAM3_MIN_MASK_AREA="100"
SAM3_PREVIEW_SLICES="100"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)           DATA_DIR="$2"; shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --h5-file)            H5_FILE="$2"; shift 2 ;;
        --algorithm)          ALGORITHM="$2"; shift 2 ;;
        --dtype)              DTYPE="$2"; shift 2 ;;
        --nsino-per-chunk)    NSINO="$2"; shift 2 ;;
        --nproj-per-chunk)    NPROJ="$2"; shift 2 ;;
        --fbp-filter)         FBP_FILTER="$2"; shift 2 ;;
        --rotation-axis)      ROTATION_AXIS="$2"; shift 2 ;;
        --remove-stripe-method) REMOVE_STRIPE_METHOD="$2"; shift 2 ;;
        --zarr-chunk-size)    ZARR_CHUNK_SIZE="$2"; shift 2 ;;
        --zarr-compression)   ZARR_COMPRESSION="$2"; shift 2 ;;
        --zarr-num-levels)    ZARR_NUM_LEVELS="$2"; shift 2 ;;
        --zarr-pixel-size-nm) ZARR_PIXEL_SIZE_NM="$2"; shift 2 ;;
        --zarr-ngpus)         ZARR_NGPUS="$2"; shift 2 ;;
        --zarr-io-workers)    ZARR_IO_WORKERS="$2"; shift 2 ;;
        --zarr-preview-slices) ZARR_PREVIEW_SLICES="$2"; shift 2 ;;
        --sam3-backend)       SAM3_BACKEND="$2"; shift 2 ;;
        --sam3-text-prompt)   SAM3_TEXT_PROMPT="$2"; shift 2 ;;
        --sam3-text-prompts)  SAM3_TEXT_PROMPTS="$2"; shift 2 ;;
        --sam3-confidence-threshold) SAM3_CONFIDENCE="$2"; shift 2 ;;
        --sam3-model-checkpoint) SAM3_MODEL_CHECKPOINT="$2"; shift 2 ;;
        --sam3-model-config)  SAM3_MODEL_CONFIG="$2"; shift 2 ;;
        --sam3-axis)          SAM3_AXIS="$2"; shift 2 ;;
        --sam3-batch-size)    SAM3_BATCH_SIZE="$2"; shift 2 ;;
        --sam3-points-per-side) SAM3_POINTS_PER_SIDE="$2"; shift 2 ;;
        --sam3-pred-iou-thresh) SAM3_PRED_IOU_THRESH="$2"; shift 2 ;;
        --sam3-stability-score-thresh) SAM3_STABILITY_SCORE_THRESH="$2"; shift 2 ;;
        --sam3-min-mask-area) SAM3_MIN_MASK_AREA="$2"; shift 2 ;;
        --sam3-preview-slices) SAM3_PREVIEW_SLICES="$2"; shift 2 ;;
        --no-tar)             shift ;;  # accepted but no-op on AWS (no tar step)
        *) echo "WARNING: Unknown argument $1"; shift ;;
    esac
done

if [[ -z "${DATA_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: --data-dir and --output-dir are required"
    exit 1
fi

TIFF_DIR="${OUTPUT_DIR}/tiff_slices"
ZARR_PATH="${OUTPUT_DIR}/reconstruction.ome.zarr"

# Stage 1/4: GPU Normalization
echo ""
echo "=== Stage 1/4: GPU Normalization ==="
echo "Start: $(date -Iseconds)"

if [[ -n "${H5_FILE}" ]]; then
    RAW_H5="${DATA_DIR}/${H5_FILE}"
else
    RAW_H5=$(ls "${DATA_DIR}"/*.h5 "${DATA_DIR}"/*.hdf5 2>/dev/null | head -n1)
    if [[ -z "${RAW_H5}" ]]; then
        echo "ERROR: No HDF5 file found in ${DATA_DIR}"
        exit 1
    fi
fi

python "${SCRIPT_DIR}/normalize_raw.py" \
    --raw-h5 "${RAW_H5}" \
    --output-dir "${OUTPUT_DIR}"
echo "Stage 1/4 complete: $(date -Iseconds)"

# Stage 2/4: GPU Reconstruction (tomocupy)
echo ""
echo "=== Stage 2/4: GPU Reconstruction ==="
echo "Start: $(date -Iseconds)"

RECON_ARGS=("--data-dir" "${OUTPUT_DIR}" "--output-dir" "${OUTPUT_DIR}")
RECON_ARGS+=("--algorithm" "${ALGORITHM}" "--dtype" "${DTYPE}")
RECON_ARGS+=("--nsino-per-chunk" "${NSINO}" "--nproj-per-chunk" "${NPROJ}")
RECON_ARGS+=("--fbp-filter" "${FBP_FILTER}")
if [[ "${ROTATION_AXIS}" != "-1.0" ]]; then
    RECON_ARGS+=("--rotation-axis" "${ROTATION_AXIS}")
fi
if [[ "${REMOVE_STRIPE_METHOD}" != "none" ]]; then
    RECON_ARGS+=("--remove-stripe-method" "${REMOVE_STRIPE_METHOD}")
fi

python "${SCRIPT_DIR}/reconstruct_gpu.py" "${RECON_ARGS[@]}"
echo "Stage 2/4 complete: $(date -Iseconds)"

# Stage 3/4: TIFF to OME-Zarr
echo ""
echo "=== Stage 3/4: TIFF to OME-Zarr ==="
echo "Start: $(date -Iseconds)"

python "${SCRIPT_DIR}/tiff_to_zarr.py" \
    --tiff-dir "${TIFF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --chunk-size "${ZARR_CHUNK_SIZE}" \
    --compression "${ZARR_COMPRESSION}" \
    --num-levels "${ZARR_NUM_LEVELS}" \
    --pixel-size-nm "${ZARR_PIXEL_SIZE_NM}" \
    --ngpus "${ZARR_NGPUS}" \
    --io-workers "${ZARR_IO_WORKERS}" \
    --preview-slices "${ZARR_PREVIEW_SLICES}" \
    --no-gpu-pyramid \
    --no-tar

echo "Stage 3/4 complete: $(date -Iseconds)"

# Stage 4/4: SAM3 Segmentation
echo ""
echo "=== Stage 4/4: SAM3 Segmentation ==="
echo "Start: $(date -Iseconds)"

SAM3_ARGS=("--zarr-path" "${ZARR_PATH}" "--output-dir" "${OUTPUT_DIR}")
SAM3_ARGS+=("--backend" "${SAM3_BACKEND}")
SAM3_ARGS+=("--confidence-threshold" "${SAM3_CONFIDENCE}")
SAM3_ARGS+=("--model-checkpoint" "${SAM3_MODEL_CHECKPOINT}")
SAM3_ARGS+=("--model-config" "${SAM3_MODEL_CONFIG}")
SAM3_ARGS+=("--axis" "${SAM3_AXIS}")
SAM3_ARGS+=("--batch-size" "${SAM3_BATCH_SIZE}")
SAM3_ARGS+=("--points-per-side" "${SAM3_POINTS_PER_SIDE}")
SAM3_ARGS+=("--pred-iou-thresh" "${SAM3_PRED_IOU_THRESH}")
SAM3_ARGS+=("--stability-score-thresh" "${SAM3_STABILITY_SCORE_THRESH}")
SAM3_ARGS+=("--min-mask-area" "${SAM3_MIN_MASK_AREA}")
SAM3_ARGS+=("--preview-slices" "${SAM3_PREVIEW_SLICES}")
SAM3_ARGS+=("--no-tar")

if [[ -n "${SAM3_TEXT_PROMPTS}" ]]; then
    SAM3_ARGS+=("--text-prompts" "${SAM3_TEXT_PROMPTS}")
else
    SAM3_ARGS+=("--text-prompt" "${SAM3_TEXT_PROMPT}")
fi

echo "Launching SAM3 via torchrun (1 node, ${NGPUS} GPUs)"
torchrun --nnodes=1 --nproc_per_node="${NGPUS}" \
    --rdzv_id=$$ --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    "${SCRIPT_DIR}/sam3_segment.py" "${SAM3_ARGS[@]}"

# Post-processing
echo ""
echo "SAM3 post-processing..."
python "${SCRIPT_DIR}/sam3_segment.py" --post-process-only "${SAM3_ARGS[@]}"

echo "Stage 4/4 complete: $(date -Iseconds)"

echo ""
echo "=============================="
echo "Full pipeline complete: $(date -Iseconds)"
echo "Output directory: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo "=============================="

exit 0
