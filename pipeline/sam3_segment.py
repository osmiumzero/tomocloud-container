#!/usr/bin/env python
"""
Multi-backend segmentation on GPU for NERSC Perlmutter.

Supports three backends:
  - threshold: Otsu + watershed (scikit-image, always available, no GPU needed)
  - sam:       SAM 2.1 automatic mask generation (requires torch + sam2)
  - sam3:      SAM 3 text-prompted detection (requires torch + sam3) [default]

Produces preview + bulk outputs (mirroring tiff_to_zarr.py):
  - segmentation_preview.zarr  (first N slices, fast transfer for viewer)
  - segmentation.zarr          (full volume)
  - segmentation.zarr.tar      (tarred full zarr for efficient bulk transfer)
  - DONE_SAM3                  (completion marker with metadata)

Multi-GPU: distributes slices across all available GPUs (4x A100 per Perlmutter
node) via torch.multiprocessing.

Usage (SAM 3, default):
    python sam3_segment.py --zarr-path /path/to/reconstruction.ome.zarr \\
        --output-dir /path/to/output

Usage (SAM 2.1):
    python sam3_segment.py --zarr-path /path/to/volume.zarr \\
        --output-dir /path/to/output --backend sam \\
        --model-checkpoint ~/models/sam2.1_hiera_large.pt

Usage (threshold, no GPU):
    python sam3_segment.py --zarr-path /path/to/volume.zarr \\
        --output-dir /path/to/output --backend threshold --single-gpu
"""

import argparse
import json
import os
import tarfile
import time
import numpy as np


def log(msg):
    """Print with flush."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_slice_to_rgb(slice_2d):
    """Convert float32 grayscale slice to uint8 RGB.

    Uses percentile-based contrast stretching (1st to 99th percentile),
    maps to 0-255 uint8, and stacks to (H, W, 3) RGB.
    """
    p1, p99 = np.percentile(slice_2d, [1, 99])
    if p99 - p1 < 1e-8:
        gray = np.full(slice_2d.shape, 128, dtype=np.uint8)
    else:
        stretched = (slice_2d - p1) / (p99 - p1)
        stretched = np.clip(stretched, 0.0, 1.0)
        gray = (stretched * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def masks_to_labels(masks, height, width):
    """Convert mask list to a single uint16 label map.

    Sorts masks by area (largest first) so smaller masks paint over larger
    ones, ensuring small features are preserved. Background remains 0.
    """
    label_map = np.zeros((height, width), dtype=np.uint16)
    if not masks:
        return label_map
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for i, mask in enumerate(sorted_masks):
        label_map[mask["segmentation"]] = i + 1
    return label_map


def get_dir_size(path):
    """Total bytes of all files under a directory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def _prompt_suffix(prompt, prompts):
    """Return output suffix for a prompt. Empty string if only 1 prompt."""
    if len(prompts) == 1:
        return ""
    safe = prompt.lower().replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return f"_{safe}"


# ---------------------------------------------------------------------------
# ThresholdBackend -- always available (scikit-image + scipy)
# ---------------------------------------------------------------------------

class ThresholdBackend:
    """Segmentation using Otsu threshold + watershed from scikit-image."""

    name = "threshold"

    def segment_slice(self, slice_2d, min_region_size=100, **kwargs):
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.filters import threshold_otsu
        from skimage.segmentation import watershed

        h, w = slice_2d.shape
        p1, p99 = np.percentile(slice_2d, [1, 99])
        if p99 - p1 < 1e-8:
            return np.zeros((h, w), dtype=np.uint16), 0
        normalized = np.clip((slice_2d - p1) / (p99 - p1), 0.0, 1.0)

        threshold = threshold_otsu(normalized)
        binary = normalized > threshold
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=10, labels=binary)
        if len(coords) == 0:
            return np.zeros((h, w), dtype=np.uint16), 0

        markers = np.zeros((h, w), dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        ws_labels = watershed(-distance, markers, mask=binary)

        # Remove small regions
        region_ids, region_counts = np.unique(ws_labels, return_counts=True)
        for rid, cnt in zip(region_ids, region_counts):
            if rid != 0 and cnt < min_region_size:
                ws_labels[ws_labels == rid] = 0

        # Relabel consecutively
        unique_ids = np.unique(ws_labels)
        unique_ids = unique_ids[unique_ids != 0]
        relabelled = np.zeros((h, w), dtype=np.uint16)
        for new_id, old_id in enumerate(unique_ids, start=1):
            relabelled[ws_labels == old_id] = new_id

        return relabelled, int(len(unique_ids))


# ---------------------------------------------------------------------------
# SAMBackend -- SAM 2.1 automatic mask generation
# ---------------------------------------------------------------------------

class SAMBackend:
    """Segmentation using Meta SAM 2.1 automatic mask generation."""

    name = "sam"

    def __init__(self, model_checkpoint, model_config="sam2.1_hiera_l",
                 device=None, points_per_side=32, pred_iou_thresh=0.7,
                 stability_score_thresh=0.8, min_mask_area=100):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        self.device = device

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        log(f"Loading SAM 2.1: config={model_config}, device={device}")
        sam2 = build_sam2(
            model_config, model_checkpoint,
            device=device, apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=points_per_side,
            points_per_batch=128,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_area,
        )
        log("SAM 2.1 model loaded.")

    def segment_slice(self, slice_2d, **kwargs):
        import torch
        h, w = slice_2d.shape
        rgb_slice = normalize_slice_to_rgb(slice_2d)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self.mask_generator.generate(rgb_slice)
        label_map = masks_to_labels(masks, h, w)
        return label_map, int(label_map.max())


# ---------------------------------------------------------------------------
# SAM3Backend -- SAM 3 text-prompted detection
# ---------------------------------------------------------------------------

class SAM3Backend:
    """Segmentation using Meta SAM 3 text-prompted detection.

    Uses a text prompt (default "circle") to detect all features in each
    slice. ~1.2s/slice for 3232x3232 on A100 -- much faster than SAM 2.1.
    """

    name = "sam3"

    def __init__(self, device=None, confidence_threshold=0.05,
                 text_prompt="circle"):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        self.device = device
        self.text_prompt = text_prompt
        self._confidence_threshold = confidence_threshold

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_str = str(device) if not isinstance(device, str) else device
        log(f"Loading SAM 3: device={device_str}")

        # build_sam3_image_model checks `device == "cuda"` internally
        builder_device = "cuda" if "cuda" in device_str else "cpu"
        model = build_sam3_image_model(
            device=builder_device, eval_mode=True,
        )
        model = model.to(device).float()

        self.processor = Sam3Processor(
            model,
            confidence_threshold=confidence_threshold,
            device=device_str,
        )
        log(f"SAM 3 model loaded (prompt='{text_prompt}', "
            f"conf_threshold={confidence_threshold}).")

    def segment_slice(self, slice_2d, text_prompt=None,
                      confidence_threshold=None, **kwargs):
        import torch
        from PIL import Image

        h, w = slice_2d.shape
        prompt = text_prompt or self.text_prompt

        if (confidence_threshold is not None
                and confidence_threshold != self._confidence_threshold):
            self._confidence_threshold = confidence_threshold
            self.processor.set_confidence_threshold(confidence_threshold)

        rgb_array = normalize_slice_to_rgb(slice_2d)
        pil_image = Image.fromarray(rgb_array, mode="RGB")

        state = self.processor.set_image(pil_image)
        state = self.processor.set_text_prompt(prompt=prompt, state=state)

        masks_tensor = state.get("masks")
        scores_tensor = state.get("scores")

        if masks_tensor is None or masks_tensor.numel() == 0:
            return np.zeros((h, w), dtype=np.uint16), 0

        masks_np = masks_tensor.squeeze(1).cpu().numpy()
        scores_np = scores_tensor.cpu().float().numpy()

        mask_list = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            area = int(m.sum())
            if area > 0:
                mask_list.append({
                    "segmentation": m,
                    "area": area,
                    "score": float(scores_np[i]),
                })

        label_map = masks_to_labels(mask_list, h, w)
        return label_map, int(label_map.max())

    def segment_slice_multi(self, slice_2d, text_prompts,
                            confidence_threshold=None, **kwargs):
        """Segment one slice with multiple text prompts, reusing image state.

        Calls set_image() once per slice, then set_text_prompt() for each
        prompt.  Returns dict of {prompt: (label_map, n_regions)}.
        """
        import torch
        from PIL import Image

        h, w = slice_2d.shape

        if (confidence_threshold is not None
                and confidence_threshold != self._confidence_threshold):
            self._confidence_threshold = confidence_threshold
            self.processor.set_confidence_threshold(confidence_threshold)

        rgb_array = normalize_slice_to_rgb(slice_2d)
        pil_image = Image.fromarray(rgb_array, mode="RGB")

        # Set image once (expensive), reuse for each prompt
        image_state = self.processor.set_image(pil_image)

        results = {}
        for prompt in text_prompts:
            state = self.processor.set_text_prompt(
                prompt=prompt, state=dict(image_state))

            masks_tensor = state.get("masks")
            scores_tensor = state.get("scores")

            if masks_tensor is None or masks_tensor.numel() == 0:
                results[prompt] = (np.zeros((h, w), dtype=np.uint16), 0)
                continue

            masks_np = masks_tensor.squeeze(1).cpu().numpy()
            scores_np = scores_tensor.cpu().float().numpy()

            mask_list = []
            for i in range(masks_np.shape[0]):
                m = masks_np[i]
                area = int(m.sum())
                if area > 0:
                    mask_list.append({
                        "segmentation": m,
                        "area": area,
                        "score": float(scores_np[i]),
                    })

            label_map = masks_to_labels(mask_list, h, w)
            results[prompt] = (label_map, int(label_map.max()))

        return results


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def create_backend(args, device=None):
    """Create the appropriate segmentation backend."""
    backend_name = args.backend

    if backend_name == "auto":
        try:
            from sam3.model_builder import build_sam3_image_model  # noqa: F401
            backend_name = "sam3"
        except ImportError:
            try:
                from sam2.build_sam import build_sam2  # noqa: F401
                backend_name = "sam"
            except ImportError:
                backend_name = "threshold"
        log(f"Auto-selected backend: {backend_name}")

    if backend_name == "threshold":
        return ThresholdBackend()
    elif backend_name == "sam":
        return SAMBackend(
            model_checkpoint=args.model_checkpoint,
            model_config=args.model_config,
            device=device,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_area=args.min_mask_area,
        )
    elif backend_name == "sam3":
        return SAM3Backend(
            device=device,
            confidence_threshold=args.confidence_threshold,
            text_prompt=args.text_prompt,
        )
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# ---------------------------------------------------------------------------
# Multi-GPU segmentation
# ---------------------------------------------------------------------------

def _gpu_worker(gpu_id, slice_indices, zarr_path, args, shared_bufs,
                output_shape, output_dtype, prompts):
    """Worker function for one GPU, supporting multiple prompts.

    Each worker loads only its slice range from zarr to limit peak memory
    (critical for Perlmutter's 256 GB nodes vs Polaris's 512 GB).

    Args:
        zarr_path: path to OME-Zarr volume (loaded per-worker)
        shared_bufs: list of mp.Array, one per prompt (parallel with prompts)
        prompts: list of text prompt strings
    """
    t_worker_start = time.perf_counter()
    import torch
    import zarr
    t_torch_import = time.perf_counter()
    device = torch.device(f"cuda:{gpu_id}")

    # Set default CUDA device to avoid buffer misplacement on multi-GPU
    torch.cuda.set_device(device)

    log(f"  GPU {gpu_id}: initializing {args.backend} on {device} "
        f"({len(slice_indices)} slices, {len(prompts)} prompt(s))")
    log(f"  GPU {gpu_id}: torch import took {t_torch_import - t_worker_start:.1f}s")

    # Load only this worker's slice range from zarr
    t_load = time.perf_counter()
    store = zarr.open(zarr_path, mode="r")
    data = store["0"] if "0" in store else store
    raw = data[0] if data.ndim > 3 else data
    start_idx, end_idx = min(slice_indices), max(slice_indices) + 1
    if args.axis == "z":
        chunk = np.array(raw[start_idx:end_idx, :, :], dtype=np.float32)
    elif args.axis == "y":
        chunk = np.array(raw[:, start_idx:end_idx, :], dtype=np.float32)
    else:
        chunk = np.array(raw[:, :, start_idx:end_idx], dtype=np.float32)
    log(f"  GPU {gpu_id}: loaded chunk {chunk.shape} "
        f"({chunk.nbytes / 1e9:.1f} GB) in {time.perf_counter() - t_load:.1f}s")

    t_model_start = time.perf_counter()
    backend = create_backend(args, device=device)
    t_model_done = time.perf_counter()
    log(f"  GPU {gpu_id}: model load took {t_model_done - t_model_start:.1f}s")

    # Create numpy views for each prompt's shared buffer
    prompt_arrays = []
    for buf in shared_bufs:
        raw_buf = buf.get_obj() if hasattr(buf, "get_obj") else buf
        arr = np.frombuffer(raw_buf, dtype=output_dtype).reshape(output_shape)
        prompt_arrays.append(arr)

    multi_prompt = len(prompts) > 1 and hasattr(backend, "segment_slice_multi")
    total_masks = {p: 0 for p in prompts}

    for count, idx in enumerate(slice_indices):
        local_idx = idx - start_idx
        if args.axis == "z":
            slice_2d = chunk[local_idx, :, :]
        elif args.axis == "y":
            slice_2d = chunk[:, local_idx, :]
        else:
            slice_2d = chunk[:, :, local_idx]

        if multi_prompt:
            results = backend.segment_slice_multi(slice_2d, prompts)
            for pi, prompt in enumerate(prompts):
                label_slice, n_regions = results[prompt]
                total_masks[prompt] += n_regions
                if args.axis == "z":
                    prompt_arrays[pi][idx, :, :] = label_slice
                elif args.axis == "y":
                    prompt_arrays[pi][:, idx, :] = label_slice
                else:
                    prompt_arrays[pi][:, :, idx] = label_slice
        else:
            label_slice, n_regions = backend.segment_slice(
                slice_2d, text_prompt=prompts[0])
            total_masks[prompts[0]] += n_regions
            if args.axis == "z":
                prompt_arrays[0][idx, :, :] = label_slice
            elif args.axis == "y":
                prompt_arrays[0][:, idx, :] = label_slice
            else:
                prompt_arrays[0][:, :, idx] = label_slice

        if (count + 1) % 50 == 0:
            log(f"  GPU {gpu_id}: {count + 1}/{len(slice_indices)} slices")

    for prompt in prompts:
        log(f"  GPU {gpu_id}: prompt '{prompt}' — "
            f"{total_masks[prompt]} regions found")


def segment_volume_multi_gpu(zarr_path, volume_shape, args, prompts):
    """Process all slices using multiple GPUs via torch.multiprocessing.

    Each worker loads only its slice range from zarr to limit peak memory.

    Args:
        zarr_path: path to OME-Zarr volume
        volume_shape: (nz, ny, nx) tuple from zarr metadata
        prompts: list of text prompt strings

    Returns:
        dict of {prompt: labels_array} where each is (nz, ny, nx) uint16
    """
    import ctypes
    import multiprocessing as mp

    nz, ny, nx = volume_shape
    axis_sizes = {"z": nz, "y": ny, "x": nx}
    n_slices = axis_sizes[args.axis]
    import torch as _torch
    num_gpus = _torch.cuda.device_count() or 1

    log(f"Multi-GPU segmentation: {num_gpus} GPUs, {n_slices} slices "
        f"along {args.axis}, backend={args.backend}, "
        f"{len(prompts)} prompt(s)")

    total_elements = nz * ny * nx
    # Allocate one shared buffer per prompt
    shared_bufs = [
        mp.Array(ctypes.c_uint16, total_elements, lock=False)
        for _ in prompts
    ]

    # Divide slices across GPUs
    base_count = n_slices // num_gpus
    remainder = n_slices % num_gpus
    slice_groups = []
    start = 0
    for g in range(num_gpus):
        count = base_count + (1 if g < remainder else 0)
        slice_groups.append(list(range(start, start + count)))
        start += count

    output_shape = (nz, ny, nx)
    output_dtype = np.dtype(np.uint16)

    # Each worker opens zarr independently and loads only its chunk.
    # Fork is safe because parent hasn't initialized CUDA.

    processes = []
    for gpu_id in range(num_gpus):
        if not slice_groups[gpu_id]:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, slice_groups[gpu_id], zarr_path, args,
                  shared_bufs, output_shape, output_dtype, prompts),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check for worker failures
    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        codes = {p.name: p.exitcode for p in failed}
        raise RuntimeError(
            f"GPU worker(s) failed: {codes}. "
            "Check stderr for CUDA OOM or model load errors."
        )

    results = {}
    for prompt, buf in zip(prompts, shared_bufs):
        results[prompt] = np.frombuffer(
            buf, dtype=np.uint16).reshape(output_shape).copy()
    return results


def segment_volume_distributed(zarr_path, args, prompts, rank, world_size,
                               local_rank):
    """Distributed segmentation across multiple nodes via torchrun.

    Each rank (= 1 GPU) processes a contiguous slice range. Results are
    written to shared zarr files on the parallel filesystem.

    Returns dict with timing breakdown: seg_time, model_load_time,
    data_load_time, write_time.  Post-processing (preview, tar, DONE) is
    handled separately by --post-process-only.
    """
    import torch
    import torch.distributed as dist
    import zarr
    import numcodecs

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ── Load volume dimensions from zarr (metadata only) ──────────────
    store = zarr.open(zarr_path, mode="r")
    data = store["0"] if "0" in store else store
    nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
    n_slices = {"z": nz, "y": ny, "x": nx}[args.axis]

    log(f"[rank {rank}] Volume: ({nz}, {ny}, {nx}), "
        f"slices={n_slices}, device={device}")

    # ── Compute slice range for this rank ─────────────────────────────
    base = n_slices // world_size
    remainder = n_slices % world_size
    start = rank * base + min(rank, remainder)
    count = base + (1 if rank < remainder else 0)
    end = start + count
    my_indices = list(range(start, end))

    log(f"[rank {rank}] Slice range: {start}-{end-1} ({count} slices)")

    # ── Load only this rank's slices ──────────────────────────────────
    t_load = time.perf_counter()
    # Handle 4D/5D OME-Zarr
    if data.ndim == 5:
        raw = data[0, 0]  # (z, y, x)
    elif data.ndim == 4:
        raw = data[0]
    else:
        raw = data

    if args.axis == "z":
        my_volume = np.array(raw[start:end, :, :], dtype=np.float32)
    elif args.axis == "y":
        my_volume = np.array(raw[:, start:end, :], dtype=np.float32)
    else:
        my_volume = np.array(raw[:, :, start:end], dtype=np.float32)

    load_time = time.perf_counter() - t_load
    log(f"[rank {rank}] Loaded {my_volume.shape} in {load_time:.1f}s "
        f"({my_volume.nbytes / 1e9:.1f} GB)")

    # ── Create output zarr stores (rank 0 creates, others wait) ───────
    chunk_size = args.chunk_size
    compressor = numcodecs.Blosc(cname="lz4", clevel=5)

    seg_zarr_paths = {}
    for prompt in prompts:
        suffix = _prompt_suffix(prompt, prompts)
        seg_zarr_paths[prompt] = os.path.join(
            args.output_dir, f"segmentation{suffix}.zarr")

    if rank == 0:
        for prompt, zpath in seg_zarr_paths.items():
            root = zarr.open_group(zpath, mode="w", zarr_format=2)
            root.create_dataset(
                "0", shape=(nz, ny, nx), dtype=np.uint16,
                chunks=(chunk_size, chunk_size, chunk_size),
                compressor=compressor, overwrite=True)
            log(f"[rank 0] Created {zpath}")

    dist.barrier()

    # ── Initialize model ──────────────────────────────────────────────
    t_model = time.perf_counter()
    backend = create_backend(args, device=device)
    model_load_time = time.perf_counter() - t_model
    log(f"[rank {rank}] Model loaded in {model_load_time:.1f}s")

    multi_prompt = (len(prompts) > 1
                    and hasattr(backend, "segment_slice_multi"))

    # ── Segment slices ────────────────────────────────────────────────
    t_seg = time.perf_counter()
    # Allocate local label arrays for this rank's slices
    local_labels = {p: np.zeros(my_volume.shape, dtype=np.uint16)
                    for p in prompts}
    total_masks = {p: 0 for p in prompts}

    for local_idx in range(count):
        if args.axis == "z":
            slice_2d = my_volume[local_idx, :, :]
        elif args.axis == "y":
            slice_2d = my_volume[:, local_idx, :]
        else:
            slice_2d = my_volume[:, :, local_idx]

        if multi_prompt:
            results = backend.segment_slice_multi(slice_2d, prompts)
            for prompt in prompts:
                label_slice, n_regions = results[prompt]
                total_masks[prompt] += n_regions
                if args.axis == "z":
                    local_labels[prompt][local_idx, :, :] = label_slice
                elif args.axis == "y":
                    local_labels[prompt][:, local_idx, :] = label_slice
                else:
                    local_labels[prompt][:, :, local_idx] = label_slice
        else:
            label_slice, n_regions = backend.segment_slice(
                slice_2d, text_prompt=prompts[0])
            total_masks[prompts[0]] += n_regions
            if args.axis == "z":
                local_labels[prompts[0]][local_idx, :, :] = label_slice
            elif args.axis == "y":
                local_labels[prompts[0]][:, local_idx, :] = label_slice
            else:
                local_labels[prompts[0]][:, :, local_idx] = label_slice

        if (local_idx + 1) % 50 == 0:
            log(f"[rank {rank}] {local_idx + 1}/{count} slices")

    seg_time = time.perf_counter() - t_seg

    for prompt in prompts:
        log(f"[rank {rank}] prompt '{prompt}': "
            f"{total_masks[prompt]} regions, {seg_time:.1f}s")

    # ── Write this rank's slices to shared zarr ───────────────────────
    t_write = time.perf_counter()
    for prompt in prompts:
        root = zarr.open_group(seg_zarr_paths[prompt], mode="r+",
                               zarr_format=2)
        arr = root["0"]
        if args.axis == "z":
            arr[start:end, :, :] = local_labels[prompt]
        elif args.axis == "y":
            arr[:, start:end, :] = local_labels[prompt]
        else:
            arr[:, :, start:end] = local_labels[prompt]

    write_time = time.perf_counter() - t_write
    log(f"[rank {rank}] Wrote labels in {write_time:.1f}s")

    del my_volume, local_labels, backend
    dist.barrier()

    # Tear down distributed group so all ranks can exit cleanly.
    # Post-processing (preview, tar, DONE) runs as a separate process
    # launched by full_compute.py — this avoids torchrun's 300s exit
    # barrier timeout.
    dist.destroy_process_group()

    return {
        "seg_time": seg_time,
        "model_load_time": model_load_time,
        "data_load_time": load_time,
        "write_time": write_time,
    }


def segment_volume_single_gpu(volume, args):
    """Process all slices sequentially on a single GPU (fallback)."""
    nz, ny, nx = volume.shape
    axis_sizes = {"z": nz, "y": ny, "x": nx}
    n_slices = axis_sizes[args.axis]

    log(f"Single-GPU segmentation: {n_slices} slices along {args.axis}, "
        f"backend={args.backend}")

    backend = create_backend(args)
    labels = np.zeros((nz, ny, nx), dtype=np.uint16)

    for idx in range(n_slices):
        if args.axis == "z":
            slice_2d = volume[idx, :, :]
        elif args.axis == "y":
            slice_2d = volume[:, idx, :]
        else:
            slice_2d = volume[:, :, idx]

        label_slice, _n_regions = backend.segment_slice(slice_2d)

        if args.axis == "z":
            labels[idx, :, :] = label_slice
        elif args.axis == "y":
            labels[:, idx, :] = label_slice
        else:
            labels[:, :, idx] = label_slice

        if (idx + 1) % 50 == 0:
            log(f"  Processed {idx + 1}/{n_slices} slices")

    return labels


# ---------------------------------------------------------------------------
# I/O: load zarr, write zarr + preview + tar
# ---------------------------------------------------------------------------

def load_volume_from_zarr(zarr_path):
    """Load the full-resolution (level 0) volume from OME-Zarr.

    Returns numpy float32 array with shape (nz, ny, nx).
    """
    import zarr

    log(f"Opening OME-Zarr: {zarr_path}")
    store = zarr.open(zarr_path, mode="r")

    if "0" in store:
        data = store["0"]
    else:
        data = store

    log(f"  Dataset shape: {data.shape}, dtype: {data.dtype}")

    t0 = time.perf_counter()
    volume = np.array(data, dtype=np.float32)
    log(f"  Loaded volume to float32 in {time.perf_counter() - t0:.1f}s")

    # Handle 4D/5D OME-Zarr (t, c, z, y, x) by squeezing leading dims
    while volume.ndim > 3:
        volume = volume[0]
    assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"

    log(f"  Volume shape: {volume.shape}")
    return volume


def _build_label_pyramid(labels, num_levels=4):
    """Build multi-scale pyramid for label volume using nearest-neighbor.

    Unlike reconstruction pyramids that use block-averaging, label pyramids
    use nearest-neighbor (stride) downsampling to preserve integer label IDs.

    Returns list of arrays: [level0 (original), level1 (2x), level2 (4x), ...]
    """
    levels = [labels]
    current = labels
    for i in range(1, num_levels):
        # Nearest-neighbor 2x downsample: just take every other voxel
        downsampled = current[::2, ::2, ::2]
        if downsampled.size == 0:
            break
        levels.append(downsampled)
        current = downsampled
        log(f"  Pyramid level {i}: {downsampled.shape}")
    return levels


def write_segmentation_zarr(labels, output_dir, chunk_size=128,
                            num_pyramid_levels=4, pixel_size_nm=0,
                            suffix=""):
    """Write full segmentation labels as OME-Zarr with pyramid levels.

    Creates a multi-scale OME-Zarr store with nearest-neighbor downsampled
    pyramid levels for efficient multi-resolution viewing in napari.
    Uses zarr_format=2 + numcodecs compressor for Polaris compatibility.

    Args:
        suffix: output name suffix (e.g., "_circle" for multi-prompt).
                Empty string for single-prompt (backward compat).
    """
    import zarr
    import numcodecs

    zarr_path = os.path.join(output_dir, f"segmentation{suffix}.zarr")
    log(f"Writing full segmentation to {zarr_path}")
    t0 = time.perf_counter()

    compressor = numcodecs.Blosc(cname="lz4", clevel=5)

    # Build pyramid levels
    log(f"  Building {num_pyramid_levels}-level pyramid (nearest-neighbor)...")
    t_pyr = time.perf_counter()
    pyramid = _build_label_pyramid(labels, num_levels=num_pyramid_levels)
    pyr_time = time.perf_counter() - t_pyr
    log(f"  Pyramid built in {pyr_time:.1f}s ({len(pyramid)} levels)")

    # Write as OME-Zarr group with levels "0", "1", "2", ...
    root = zarr.open_group(zarr_path, mode="w", zarr_format=2)

    datasets = []
    scale = pixel_size_nm if pixel_size_nm > 0 else 1.0
    for i, level_data in enumerate(pyramid):
        level_data = level_data.astype(np.uint16)
        arr = root.create_dataset(
            str(i),
            shape=level_data.shape,
            dtype=level_data.dtype,
            chunks=(chunk_size, chunk_size, chunk_size),
            compressor=compressor,
            overwrite=True,
        )
        arr[:] = level_data
        level_scale = scale * (2 ** i)
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [level_scale, level_scale, level_scale]},
            ],
        })

    # Write OME-Zarr v0.4 metadata
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": datasets,
        "type": "nearest_neighbor_2x2x2",
    }]

    num_unique = int(np.count_nonzero(np.bincount(labels.ravel())))
    root.attrs["num_unique_labels"] = num_unique
    root.attrs["total_masks"] = int(labels.max())

    elapsed = time.perf_counter() - t0
    seg_size = get_dir_size(zarr_path)
    log(f"  Written in {elapsed:.1f}s, {seg_size / (1024**2):.1f} MB")
    log(f"  Shape: {labels.shape}, unique labels: {num_unique}, "
        f"pyramid levels: {len(pyramid)}")

    return zarr_path, seg_size


def write_preview_zarr(labels, output_dir, n_preview, chunk_size=128,
                       suffix=""):
    """Write preview segmentation (middle N slices) as Zarr.

    Extracts slices from the middle of the volume for more representative
    preview data.  Uses zarr_format=2 + numcodecs compressor for Polaris
    compatibility.

    Args:
        suffix: output name suffix (e.g., "_circle" for multi-prompt).
    """
    import zarr
    import numcodecs

    nz, ny, nx = labels.shape
    n_preview = min(n_preview, nz)
    # Extract from the MIDDLE of the volume
    mid_start = (nz - n_preview) // 2
    preview_path = os.path.join(
        output_dir, f"segmentation_preview{suffix}.zarr")
    log(f"Creating segmentation_preview.zarr "
        f"({n_preview} of {nz} slices, z={mid_start}..{mid_start + n_preview})...")
    t0 = time.perf_counter()

    compressor = numcodecs.Blosc(cname="lz4", clevel=5)
    preview_store = zarr.open_array(
        preview_path, mode="w", zarr_format=2,
        shape=(n_preview, ny, nx),
        chunks=(min(n_preview, chunk_size), chunk_size, chunk_size),
        dtype=np.uint16,
        compressor=compressor,
    )

    chunk_z = chunk_size
    for z_start in range(0, n_preview, chunk_z):
        z_end = min(z_start + chunk_z, n_preview)
        src_start = mid_start + z_start
        src_end = mid_start + z_end
        preview_store[z_start:z_end] = labels[src_start:src_end]

    # Store the source offset so the viewer knows which region this covers
    preview_store.attrs["preview_z_start"] = int(mid_start)
    preview_store.attrs["preview_z_end"] = int(mid_start + n_preview)
    preview_store.attrs["full_z_size"] = int(nz)

    preview_time = time.perf_counter() - t0
    preview_size = get_dir_size(preview_path)
    log(f"  Preview: {n_preview} slices, {preview_size / (1024**2):.1f} MB, "
        f"{preview_time:.1f}s")

    return preview_path, n_preview, preview_size


def create_tar(zarr_path, output_dir):
    """Tar the full segmentation zarr for efficient bulk transfer."""
    zarr_basename = os.path.basename(zarr_path)
    tar_filename = f"{zarr_basename}.tar"
    tar_path = os.path.join(output_dir, tar_filename)
    log(f"Creating {tar_filename}...")
    t0 = time.perf_counter()

    with tarfile.open(tar_path, "w") as tar:
        tar.add(zarr_path, arcname=zarr_basename)

    tar_size = os.path.getsize(tar_path)
    tar_time = time.perf_counter() - t0
    log(f"  Tar: {tar_size / (1024**2):.1f} MB in {tar_time:.1f}s")

    return tar_path, tar_size


def run_post_processing(labels_dict, prompts, output_dir, preview_slices,
                        chunk_size, backend, axis, seg_time, n_slices,
                        t_total, tlog_fn, is_distributed=False, args=None):
    """Write preview zarrs, tars, and DONE marker for all prompts.

    Args:
        labels_dict: {prompt: np.ndarray} mapping.
        is_distributed: if True, skip write_segmentation_zarr (already on disk).
        args: full argparse namespace (for backward-compat DONE fields).
    """
    first_labels = None
    all_tar_paths = []

    for prompt in prompts:
        labels = labels_dict[prompt]
        suffix = _prompt_suffix(prompt, prompts)

        unique_labels = int(np.count_nonzero(np.bincount(labels.ravel())))
        log(f"Prompt '{prompt}'{suffix}: {unique_labels} unique labels")

        tlog_fn(f"Preview zarr starting (prompt='{prompt}')")
        preview_path, n_preview, preview_size = write_preview_zarr(
            labels, output_dir, preview_slices, chunk_size, suffix=suffix)
        tlog_fn(f"Preview zarr done (prompt='{prompt}')")

        if is_distributed:
            zarr_path_out = os.path.join(
                output_dir, f"segmentation{suffix}.zarr")
            zarr_size = get_dir_size(zarr_path_out)
            log(f"  Distributed zarr already at {zarr_path_out} "
                f"({zarr_size / 1e6:.0f} MB)")
        else:
            tlog_fn(f"Full zarr write starting (prompt='{prompt}')")
            zarr_path_out, zarr_size = write_segmentation_zarr(
                labels, output_dir, chunk_size, suffix=suffix)
            tlog_fn(f"Full zarr write done (prompt='{prompt}')")

        no_tar = getattr(args, 'no_tar', False) if args else False
        if not no_tar:
            tlog_fn(f"Tar creation starting (prompt='{prompt}')")
            tar_path, tar_size = create_tar(zarr_path_out, output_dir)
            tlog_fn(f"Tar creation done (prompt='{prompt}')")
            all_tar_paths.append(tar_path)
        else:
            tar_path = None
            tar_size = 0
            log(f"  Skipping tar (--no-tar)")

        if first_labels is None:
            first_labels = labels
            first_unique = unique_labels
            first_preview_size = preview_size
            first_zarr_size = zarr_size
            first_tar_size = tar_size
            first_n_preview = n_preview
            first_tar_path = tar_path

    # ── Completion marker ─────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    avg_per_slice = seg_time / max(n_slices, 1)

    done_info = {
        "status": "complete",
        "stage": "sam3_segmentation",
        "backend": backend,
        "prompts": prompts,
        "shape": list(first_labels.shape),
        "axis": axis,
        "num_slices_processed": n_slices,
        "total_time_s": round(total_time, 2),
        "seg_time_s": round(seg_time, 2),
        "avg_time_per_slice_s": round(avg_per_slice, 2),
    }

    per_prompt = {}
    for prompt in prompts:
        labels = labels_dict[prompt]
        suffix = _prompt_suffix(prompt, prompts)
        unique = int(np.count_nonzero(np.bincount(labels.ravel())))
        per_prompt[prompt] = {
            "unique_labels": unique,
            "segmentation_path": f"segmentation{suffix}.zarr",
            "preview_path": f"segmentation_preview{suffix}.zarr",
            "tar_path": f"segmentation{suffix}.zarr.tar" if all_tar_paths else None,
        }
    done_info["per_prompt"] = per_prompt

    if len(prompts) == 1 and args is not None:
        done_info["text_prompt"] = prompts[0]
        done_info["confidence_threshold"] = args.confidence_threshold
        done_info["segmentation_path"] = "segmentation.zarr"
        done_info["preview_path"] = "segmentation_preview.zarr"
        done_info["tar_path"] = os.path.basename(first_tar_path) if first_tar_path else None
        done_info["unique_labels"] = first_unique
        done_info["preview_slices"] = first_n_preview
        done_info["preview_size_bytes"] = first_preview_size
        done_info["zarr_size_bytes"] = first_zarr_size
        done_info["tar_size_bytes"] = first_tar_size

    if backend == "sam" and args is not None:
        done_info["model"] = args.model_config
        done_info["points_per_side"] = args.points_per_side

    done_path = os.path.join(output_dir, "DONE_SAM3")
    with open(done_path, "w") as f:
        json.dump(done_info, f, indent=2)

    log(f"\nDone. Total: {total_time:.1f}s (segmentation: {seg_time:.1f}s, "
        f"avg {avg_per_slice:.2f}s/slice)")
    log(f"Prompts: {prompts}")
    for prompt in prompts:
        unique = int(np.count_nonzero(np.bincount(labels_dict[prompt].ravel())))
        log(f"  '{prompt}': {unique} labels")
    log(f"Output: {output_dir}")
    tlog_fn(f"Script complete. Total: {total_time:.1f}s")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-backend segmentation on GPU"
    )
    parser.add_argument("--zarr-path", required=True,
                        help="Path to OME-Zarr volume (level 0 will be read)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write segmentation outputs")
    parser.add_argument("--backend", default="sam3",
                        choices=["threshold", "sam", "sam3", "auto"],
                        help="Segmentation backend (default: sam3)")

    # SAM 2.1 options
    parser.add_argument("--model-checkpoint",
                        default="/global/u1/t/timjdunn/models/sam2.1_hiera_large.pt",
                        help="Path to SAM 2.1 model weights")
    parser.add_argument("--model-config", default="sam2.1_hiera_l",
                        help="SAM 2 config name")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="Grid density for SAM 2.1 mask generation")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7,
                        help="IoU threshold for SAM 2.1")
    parser.add_argument("--stability-score-thresh", type=float, default=0.8,
                        help="Stability score threshold for SAM 2.1")
    parser.add_argument("--min-mask-area", type=int, default=100,
                        help="Minimum mask area in pixels")

    # SAM 3 options
    parser.add_argument("--text-prompt", default="circle",
                        help="Text prompt for SAM 3 detection (default: circle)")
    parser.add_argument("--text-prompts", default="",
                        help="Comma-separated list of text prompts for "
                        "multi-prompt SAM3 (e.g., 'circle,round particle'). "
                        "Overrides --text-prompt when specified.")
    parser.add_argument("--confidence-threshold", type=float, default=0.05,
                        help="Confidence threshold for SAM 3 (default: 0.05)")

    # Processing options
    parser.add_argument("--axis", default="z", choices=["z", "y", "x"],
                        help="Slicing axis (default: z)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of GPUs to use in parallel (default: 4)")
    parser.add_argument("--single-gpu", action="store_true", default=False,
                        help="Process sequentially on a single GPU")
    parser.add_argument("--preview-slices", type=int, default=100,
                        help="Number of Z slices for preview zarr (default: 100)")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Zarr chunk size (default: 128)")
    parser.add_argument("--post-process-only", action="store_true",
                        default=False,
                        help="Skip segmentation; run only post-processing "
                        "(preview zarr, tar, DONE marker) from existing "
                        "segmentation zarrs on disk.")
    parser.add_argument("--no-tar", action="store_true",
                        help="Skip creating .tar archives of segmentation zarrs")

    return parser.parse_args()


def _init_distributed():
    """Detect and initialize distributed training (torchrun).

    Returns (rank, world_size, local_rank).  When not running under
    torchrun, returns (0, 1, 0) — equivalent to single-node mode.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        import torch
        import torch.distributed as dist
        dist.init_process_group(backend="gloo")
        torch.cuda.set_device(local_rank)
        log(f"Distributed: rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}")
    return rank, world_size, local_rank


def main():
    t_script_start = time.perf_counter()
    args = parse_args()
    t_total = time.perf_counter()

    # ── Resolve prompt list ──────────────────────────────────────────
    if args.text_prompts:
        prompts = [p.strip() for p in args.text_prompts.split(",") if p.strip()]
    else:
        prompts = [args.text_prompt]

    if len(prompts) > 1 and args.backend != "sam3":
        raise ValueError(
            f"Multi-prompt requires SAM3 backend, got '{args.backend}'")

    # ── Distributed mode detection ───────────────────────────────────
    rank, world_size, local_rank = _init_distributed()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Timing log (written to file to avoid stdout truncation) ───────
    timing_log_path = os.path.join(args.output_dir, "sam3_timing.log")
    def tlog(label):
        elapsed = time.perf_counter() - t_script_start
        msg = f"[{elapsed:8.1f}s] {label}"
        log(msg)
        with open(timing_log_path, "a") as f:
            f.write(msg + "\n")

    tlog("Script started (argparse done)")
    log(f"Prompts: {prompts}")

    # ── Post-process-only mode (runs after distributed torchrun) ──────
    if args.post_process_only:
        tlog("Post-process-only mode")
        import zarr

        summary_path = os.path.join(args.output_dir, "seg_summary.json")
        with open(summary_path) as f:
            seg_summary = json.load(f)
        seg_time = seg_summary["seg_time"]
        nz, ny, nx = seg_summary["shape"]
        n_slices = {"z": nz, "y": ny, "x": nx}[args.axis]
        tlog(f"seg_summary: seg_time={seg_time:.1f}s, "
             f"shape=({nz},{ny},{nx})")

        labels_dict = {}
        for prompt in prompts:
            suffix = _prompt_suffix(prompt, prompts)
            seg_zarr = os.path.join(
                args.output_dir, f"segmentation{suffix}.zarr")
            tlog(f"Loading {seg_zarr}")
            root = zarr.open_group(seg_zarr, mode="r", zarr_format=2)
            labels_dict[prompt] = np.array(root["0"], dtype=np.uint16)
            tlog(f"Loaded: shape={labels_dict[prompt].shape}")

        run_post_processing(
            labels_dict=labels_dict, prompts=prompts,
            output_dir=args.output_dir,
            preview_slices=args.preview_slices,
            chunk_size=args.chunk_size,
            backend=args.backend, axis=args.axis,
            seg_time=seg_time, n_slices=n_slices,
            t_total=t_total, tlog_fn=tlog,
            is_distributed=True, args=args)
        return

    # ── Distributed vs single-node path ─────────────────────────────
    if world_size > 1:
        # ── Multi-node distributed (torchrun) ─────────────────────────
        tlog(f"Distributed mode: rank={rank}, world_size={world_size}, "
             f"local_rank={local_rank}")
        t_seg = time.perf_counter()
        timing = segment_volume_distributed(
            args.zarr_path, args, prompts, rank, world_size, local_rank)
        seg_time_wall = time.perf_counter() - t_seg
        seg_time = timing["seg_time"]
        tlog(f"Segmentation done: {seg_time_wall:.1f}s "
             f"(model_load={timing['model_load_time']:.1f}s, "
             f"data_load={timing['data_load_time']:.1f}s, "
             f"seg={timing['seg_time']:.1f}s, "
             f"write={timing['write_time']:.1f}s)")

        # Rank 0 writes seg_summary.json for post-processing
        if rank == 0:
            import zarr as _zarr
            _store = _zarr.open(args.zarr_path, mode="r")
            _data = _store["0"] if "0" in _store else _store
            nz, ny, nx = _data.shape[-3], _data.shape[-2], _data.shape[-1]
            seg_summary = {
                "seg_time": seg_time,
                "model_load_time": timing["model_load_time"],
                "data_load_time": timing["data_load_time"],
                "write_time": timing["write_time"],
                "world_size": world_size,
                "prompts": prompts,
                "shape": [int(nz), int(ny), int(nx)],
            }
            summary_path = os.path.join(args.output_dir, "seg_summary.json")
            with open(summary_path, "w") as f:
                json.dump(seg_summary, f, indent=2)
            tlog("Wrote seg_summary.json")

        # ALL ranks exit — post-processing runs as separate process
        log(f"[rank {rank}] Distributed segmentation complete. Exiting.")
        return
    else:
        # ── Single-node (fork multiprocessing) ────────────────────────
        # Read volume shape from zarr metadata without loading full volume.
        # Workers load only their chunk to stay within 256 GB node memory.
        import zarr as _zarr
        _store = _zarr.open(args.zarr_path, mode="r")
        _data = _store["0"] if "0" in _store else _store
        _raw = _data[0] if _data.ndim > 3 else _data
        nz, ny, nx = _raw.shape[-3], _raw.shape[-2], _raw.shape[-1]
        tlog(f"Volume shape: ({nz}, {ny}, {nx}), "
             f"{nz * ny * nx * 4 / 1e9:.1f} GB (lazy, not preloaded)")
        del _store, _data, _raw

        axis_sizes = {"z": nz, "y": ny, "x": nx}
        n_slices = axis_sizes[args.axis]

        log(f"Segmentation: backend={args.backend}, axis={args.axis}, "
            f"slices={n_slices}, prompts={len(prompts)}")

        tlog("Segmentation starting (multi-GPU fork)")
        t_seg = time.perf_counter()
        if args.single_gpu:
            # Single-GPU: read slices from zarr one at a time
            _store = _zarr.open(args.zarr_path, mode="r")
            _data = _store["0"] if "0" in _store else _store
            _raw = _data[0] if _data.ndim > 3 else _data
            labels_dict = {}
            backend = create_backend(args)
            for prompt in prompts:
                labels = np.zeros((nz, ny, nx), dtype=np.uint16)
                for idx in range(n_slices):
                    if args.axis == "z":
                        slice_2d = np.array(_raw[idx, :, :], dtype=np.float32)
                    elif args.axis == "y":
                        slice_2d = np.array(_raw[:, idx, :], dtype=np.float32)
                    else:
                        slice_2d = np.array(_raw[:, :, idx], dtype=np.float32)
                    label_slice, _ = backend.segment_slice(
                        slice_2d, text_prompt=prompt)
                    if args.axis == "z":
                        labels[idx, :, :] = label_slice
                    elif args.axis == "y":
                        labels[:, idx, :] = label_slice
                    else:
                        labels[:, :, idx] = label_slice
                    if (idx + 1) % 50 == 0:
                        log(f"  Processed {idx + 1}/{n_slices} slices "
                            f"(prompt='{prompt}')")
                labels_dict[prompt] = labels
            del _store, _data, _raw
        else:
            labels_dict = segment_volume_multi_gpu(
                args.zarr_path, (nz, ny, nx), args, prompts)
        seg_time = time.perf_counter() - t_seg
        tlog(f"Segmentation done: {seg_time:.1f}s")

    # ── Write outputs (preview zarr, full zarr, tar, DONE marker) ───
    run_post_processing(
        labels_dict=labels_dict, prompts=prompts,
        output_dir=args.output_dir,
        preview_slices=args.preview_slices,
        chunk_size=args.chunk_size,
        backend=args.backend, axis=args.axis,
        seg_time=seg_time, n_slices=n_slices,
        t_total=t_total, tlog_fn=tlog,
        is_distributed=False, args=args)


if __name__ == "__main__":
    main()
