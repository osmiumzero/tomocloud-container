#!/usr/bin/env python
"""
GPU-accelerated TIFF stack to OME-Zarr conversion for NERSC Perlmutter.

Optimised for 50 GB+ datasets on Perlmutter nodes (4x A100 40 GB each).
Key improvements over the baseline version:

  * **Multi-GPU slab distribution** -- each A100 processes a Z-slab of the
    volume in its own CUDA stream.  2x2x2 block-average downsampling runs
    entirely on GPU; results are copied back per-slab so host RAM stays low.
  * **Threaded TIFF I/O** -- reads from GPFS overlap via a ThreadPoolExecutor.
  * **Threaded zarr chunk writing** -- blosc compression releases the GIL,
    so multiple chunks compress + flush to GPFS simultaneously.
  * **VRAM-aware fallback** -- if a GPU slab exceeds available VRAM the
    script gracefully processes it on CPU instead.
  * **Strided-slice downsampling** -- uses explicit addition of the 8
    sub-voxels via stride-2 slicing.  This avoids CuPy NVRTC JIT
    compilation entirely (all ops use pre-compiled elementwise kernels).

Usage (single-node, all 4 GPUs -- recommended for zarr conversion):
    python tiff_to_zarr.py --tiff-dir /path/to/tiff_slices --output-dir /out

    # CPU-only fallback:
    python tiff_to_zarr.py --tiff-dir /path/to/slices --output-dir /out --no-gpu

    # Control GPU count and I/O parallelism:
    python tiff_to_zarr.py --tiff-dir /path/to/slices --output-dir /out \
        --ngpus 4 --io-workers 8
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# GPU setup
# =====================================================================

_cupy_available: Optional[bool] = None
_cp = None  # cupy module reference


def _try_import_cupy():
    """Import CuPy once, cache the result."""
    global _cupy_available, _cp
    if _cupy_available is not None:
        return _cupy_available
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        _cp = cp
        _cupy_available = True
    except Exception:
        _cupy_available = False
    return _cupy_available


def _get_gpu_count() -> int:
    """Return the number of visible CUDA devices."""
    if not _try_import_cupy():
        return 0
    return _cp.cuda.runtime.getDeviceCount()


def _get_vram_free(device_id: int = 0) -> int:
    """Return free VRAM in bytes for *device_id*."""
    if not _try_import_cupy():
        return 0
    with _cp.cuda.Device(device_id):
        free, _total = _cp.cuda.runtime.memGetInfo()
        return free


def log(msg):
    print(msg, flush=True)


# =====================================================================
# I/O helpers
# =====================================================================

def _read_one_tiff(fpath: str) -> np.ndarray:
    """Read a single TIFF file as float32.  Thread-safe."""
    import tifffile
    return tifffile.imread(fpath).astype(np.float32)


def load_tiff_stack(tiff_dir: str, io_workers: int = 8) -> np.ndarray:
    """Read all ``slice_*.tif`` files into a (nz, ny, nx) float32 array.

    Uses *io_workers* threads to overlap reads from the parallel
    filesystem (GPFS / Eagle).
    """
    import tifffile

    files = sorted(glob.glob(os.path.join(tiff_dir, "slice_*.tif")))
    if not files:
        files = sorted(glob.glob(os.path.join(tiff_dir, "recon_*.tiff")))
    if not files:
        raise FileNotFoundError(f"No slice_*.tif or recon_*.tiff files found in {tiff_dir}")

    # Dimensions from first slice
    first = tifffile.imread(files[0])
    ny, nx = first.shape[:2]
    nz = len(files)
    log(f"  Found {nz} TIFF slices, volume shape: ({nz}, {ny}, {nx})")

    volume = np.empty((nz, ny, nx), dtype=np.float32)
    volume[0] = first.astype(np.float32)

    # Parallel reads
    def _load(idx_path):
        idx, fp = idx_path
        volume[idx] = _read_one_tiff(fp)
        return idx

    loaded = 1
    report_every = max(1, nz // 20)

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {pool.submit(_load, (i, f)): i
                   for i, f in enumerate(files[1:], start=1)}
        for fut in as_completed(futures):
            fut.result()
            loaded += 1
            if loaded % report_every == 0 or loaded == nz:
                log(f"  Loaded {loaded}/{nz} slices")

    return volume


# =====================================================================
# Downsample (works with both numpy and CuPy arrays)
# =====================================================================

def _downsample_block_avg(vol):
    """Downsample a 3D volume by 2x via 2x2x2 block averaging.

    Uses explicit strided-slice addition of the 8 sub-voxels.
    Compatible with both numpy and CuPy arrays (duck-typed).
    Avoids reshape + .mean() which triggers CuPy NVRTC JIT.
    """
    nz, ny, nx = vol.shape
    nz2 = (nz // 2) * 2
    ny2 = (ny // 2) * 2
    nx2 = (nx // 2) * 2
    v = vol[:nz2, :ny2, :nx2]

    d = (
        v[0::2, 0::2, 0::2] + v[0::2, 0::2, 1::2] +
        v[0::2, 1::2, 0::2] + v[0::2, 1::2, 1::2] +
        v[1::2, 0::2, 0::2] + v[1::2, 0::2, 1::2] +
        v[1::2, 1::2, 0::2] + v[1::2, 1::2, 1::2]
    )
    d /= 8.0
    return d


def _to_numpy(arr) -> np.ndarray:
    """Move a CuPy array to host, or return numpy array unchanged."""
    if hasattr(arr, "__cuda_array_interface__") and _cp is not None:
        return _cp.asnumpy(arr)
    return np.asarray(arr)


# =====================================================================
# Multi-GPU pyramid builder
# =====================================================================

def _build_slab_pyramid_gpu(
    slab_np: np.ndarray,
    device_id: int,
    num_levels: int,
) -> List[np.ndarray]:
    """Build a multiscale pyramid for *slab_np* on GPU *device_id*.

    Returns a list of numpy arrays (one per level).
    Falls back to CPU if the slab doesn't fit in VRAM.
    """
    levels: List[np.ndarray] = [slab_np]

    # Check VRAM: need slab (1x) + downsample output (1/8x) + temp (1/8x)
    # ~1.25x is the true peak; use 1.5x for safety margin
    vram_free = _get_vram_free(device_id)
    vram_needed = int(slab_np.nbytes * 1.5)

    if vram_free >= vram_needed:
        # GPU path
        slab_gb = slab_np.nbytes / 1024**3
        free_gb = vram_free / 1024**3
        log(f"    GPU:{device_id} slab {slab_gb:.1f} GB "
            f"(need {vram_needed / 1024**3:.1f} GB) — "
            f"{free_gb:.1f} GB free — using GPU")
        with _cp.cuda.Device(device_id):
            current = _cp.asarray(slab_np)
            for lvl in range(1, num_levels):
                if min(current.shape) < 2:
                    break
                current = _downsample_block_avg(current)
                _cp.cuda.Stream.null.synchronize()
                levels.append(_cp.asnumpy(current))
            del current
            _cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback for this slab
        slab_gb = slab_np.nbytes / 1024**3
        free_gb = vram_free / 1024**3
        log(f"    GPU:{device_id} slab {slab_gb:.1f} GB "
            f"(need {vram_needed / 1024**3:.1f} GB) — "
            f"only {free_gb:.1f} GB free — using CPU")
        current = slab_np
        for lvl in range(1, num_levels):
            if min(current.shape) < 2:
                break
            current = _downsample_block_avg(current)
            levels.append(np.array(current))

    return levels


def build_multiscale_pyramid(
    volume: np.ndarray,
    num_levels: int,
    use_gpu: bool = True,
    ngpus: int = 4,
) -> List[np.ndarray]:
    """Build a multiscale pyramid, distributing Z-slabs across GPUs.

    For large volumes (> 1 GPU VRAM), the volume is split along Z into
    *ngpus* slabs.  Each GPU builds its own sub-pyramid in parallel
    (via threads — CuPy releases the GIL for kernel launches).  The
    sub-pyramids are concatenated along Z at each level.

    Level 0 is always the original *volume* (no copy).

    Parameters
    ----------
    volume : np.ndarray
        Full volume on host, shape (nz, ny, nx), float32.
    num_levels : int
        Number of pyramid levels including the original.
    use_gpu : bool
        Attempt GPU-accelerated downsampling (with CPU fallback).
    ngpus : int
        Number of GPUs to distribute work across (default 4).

    Returns
    -------
    list of np.ndarray
        Pyramid from finest (index 0) to coarsest.
    """
    nz = volume.shape[0]
    log(f"  Level 0: {volume.shape}")

    if not use_gpu or not _try_import_cupy():
        # Pure CPU path
        pyramid = [volume]
        current = volume
        for lvl in range(1, num_levels):
            if min(current.shape) < 2:
                log(f"  Stopping at level {lvl} (dimension too small)")
                break
            t0 = time.perf_counter()
            current = _downsample_block_avg(current)
            dt = time.perf_counter() - t0
            pyramid.append(np.array(current))
            log(f"  Level {lvl}: {pyramid[-1].shape}  [CPU, {dt:.3f}s]")
        return pyramid

    # Multi-GPU path: split volume into Z-slabs
    actual_gpus = min(ngpus, _get_gpu_count(), nz)
    if actual_gpus < 1:
        actual_gpus = 1
    log(f"  Using {actual_gpus} GPU(s) for pyramid construction")

    # Compute slab boundaries (balanced)
    slab_boundaries = []
    base = nz // actual_gpus
    remainder = nz % actual_gpus
    start = 0
    for g in range(actual_gpus):
        count = base + (1 if g < remainder else 0)
        slab_boundaries.append((start, start + count))
        start += count

    # Build sub-pyramids in parallel (one thread per GPU)
    from concurrent.futures import ThreadPoolExecutor

    sub_pyramids: List[Optional[List[np.ndarray]]] = [None] * actual_gpus

    def _build_gpu(g):
        z0, z1 = slab_boundaries[g]
        slab = volume[z0:z1]
        log(f"    GPU:{g} slab [{z0}:{z1}] = {slab.shape}")
        sub_pyramids[g] = _build_slab_pyramid_gpu(slab, g, num_levels)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=actual_gpus) as pool:
        list(pool.map(_build_gpu, range(actual_gpus)))
    dt = time.perf_counter() - t0

    # Concatenate sub-pyramids along Z for each level
    # Level 0 = original volume (no concat needed)
    pyramid = [volume]

    # Determine how many levels all sub-pyramids produced
    min_sub_levels = min(len(sp) for sp in sub_pyramids)

    for lvl in range(1, min_sub_levels):
        parts = [sub_pyramids[g][lvl] for g in range(actual_gpus)]
        combined = np.concatenate(parts, axis=0)
        pyramid.append(combined)
        log(f"  Level {lvl}: {combined.shape}  "
            f"[{actual_gpus} GPUs, {dt:.3f}s total]")

    return pyramid


# =====================================================================
# Zarr writer with threaded chunk I/O
# =====================================================================

def write_ome_zarr(
    pyramid: List[np.ndarray],
    output_path: str,
    chunk_size: int,
    compression: str,
    pixel_size_nm: float,
    io_workers: int = 8,
) -> None:
    """Write a multiscale pyramid as OME-Zarr v0.4.

    Uses direct zarr array creation + parallel chunk writing via a
    thread pool (blosc compression releases the GIL).

    Parameters
    ----------
    pyramid : list of np.ndarray
        Pyramid levels, finest first.
    output_path : str
        Path for the ``.ome.zarr`` directory store.
    chunk_size : int
        Cubic chunk edge length.
    compression : str
        ``"blosc-lz4"``, ``"blosc-zstd"``, or ``"none"``.
    pixel_size_nm : float
        Isotropic voxel size in nm (0 = unit scale 1.0).
    io_workers : int
        Thread count for parallel chunk writing.
    """
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.format import FormatV04

    # Compression
    compressor = None
    if compression == "blosc-lz4":
        import numcodecs
        compressor = numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    elif compression == "blosc-zstd":
        import numcodecs
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Zarr store
    store = parse_url(output_path, mode="w", fmt=FormatV04()).store
    root = zarr.group(store=store, overwrite=True, zarr_format=2)

    chunks = (chunk_size, chunk_size, chunk_size)
    ps = pixel_size_nm if pixel_size_nm > 0 else 1.0
    num_levels = len(pyramid)

    # Create arrays
    zarr_arrays = []
    for lvl, data in enumerate(pyramid):
        arr = root.create_dataset(
            str(lvl), shape=data.shape, dtype=data.dtype,
            chunks=chunks, compressor=compressor, overwrite=True)
        zarr_arrays.append(arr)

    # OME metadata
    coordinate_transformations = []
    for i in range(num_levels):
        scale = ps * (2 ** i)
        coordinate_transformations.append(
            [{"type": "scale", "scale": [scale, scale, scale]}])

    axes = [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "",
        "axes": axes,
        "datasets": [
            {"path": str(i),
             "coordinateTransformations": coordinate_transformations[i]}
            for i in range(num_levels)
        ],
        "type": "block_average_2x2x2",
    }]

    # Write each level with threaded chunk I/O
    cs = chunk_size
    for lvl in range(num_levels):
        data = pyramid[lvl]
        arr = zarr_arrays[lvl]
        nz, ny, nx = data.shape

        log(f"  Writing level {lvl} {data.shape} "
            f"({data.nbytes / 1024**2:.0f} MB)")

        # Small levels: direct write is faster
        if data.nbytes < 64 * 1024 * 1024:
            arr[:] = data
            continue

        # Build chunk slices
        chunk_slices = []
        for z0 in range(0, nz, cs):
            for y0 in range(0, ny, cs):
                for x0 in range(0, nx, cs):
                    slc = (
                        slice(z0, min(z0 + cs, nz)),
                        slice(y0, min(y0 + cs, ny)),
                        slice(x0, min(x0 + cs, nx)),
                    )
                    chunk_slices.append(slc)

        def _write_chunk(slc):
            arr[slc] = data[slc]

        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            list(pool.map(_write_chunk, chunk_slices))

    store.close()
    log(f"  Wrote OME-Zarr v0.4: {output_path}")


# =====================================================================
# Utilities
# =====================================================================

def get_dir_size(path: str) -> int:
    """Walk a directory tree and return total size in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fpath)
            except OSError:
                pass
    return total


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TIFF stack to OME-Zarr conversion (multi-GPU, NERSC Perlmutter)"
    )
    parser.add_argument("--tiff-dir", required=True,
                        help="Path to directory with slice_NNNN.tif files")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write reconstruction.ome.zarr/")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Cubic chunk edge length (default: 128)")
    parser.add_argument("--compression", default="blosc-lz4",
                        choices=["blosc-lz4", "blosc-zstd", "none"],
                        help="Compression codec (default: blosc-lz4)")
    parser.add_argument("--num-levels", type=int, default=4,
                        help="Number of multiscale pyramid levels (default: 4)")
    parser.add_argument("--pixel-size-nm", type=float, default=0,
                        help="Voxel size in nm for OME metadata (default: 0)")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="Use CuPy for GPU-accelerated downsampling (default)")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="Disable GPU, use CPU-only numpy")
    parser.add_argument("--no-gpu-pyramid", action="store_true",
                        help="Use CPU for pyramid building even when GPU is available for I/O")
    parser.add_argument("--ngpus", type=int, default=4,
                        help="Number of GPUs to use for pyramid building (default: 4)")
    parser.add_argument("--io-workers", type=int, default=8,
                        help="I/O threads for TIFF reads and zarr writes (default: 8)")
    parser.add_argument("--no-tar", action="store_true",
                        help="Skip creating .tar archive of the zarr")
    parser.add_argument("--preview-slices", type=int, default=100,
                        help="Number of Z slices for preview.zarr (default: 100)")
    return parser.parse_args()


def main():
    args = parse_args()

    tiff_dir = args.tiff_dir
    zarr_path = os.path.join(args.output_dir, "reconstruction.ome.zarr")
    os.makedirs(args.output_dir, exist_ok=True)

    pyramid_gpu = args.use_gpu and not args.no_gpu_pyramid
    gpu_str = f"GPU (CuPy, {args.ngpus} GPUs)" if args.use_gpu else "CPU (numpy)"
    pyr_str = "GPU" if pyramid_gpu else "CPU"
    log(f"TIFF-to-Zarr conversion: {gpu_str}, pyramid={pyr_str}, "
        f"levels={args.num_levels}, chunk={args.chunk_size}, "
        f"compression={args.compression}, io_workers={args.io_workers}")
    t_total = time.perf_counter()

    # ── Load TIFF stack ───────────────────────────────────────────────
    log("Loading TIFF stack...")
    t0 = time.perf_counter()
    volume = load_tiff_stack(tiff_dir, io_workers=args.io_workers)
    load_time = time.perf_counter() - t0
    log(f"  Load time: {load_time:.2f}s")

    # ── Build multiscale pyramid ──────────────────────────────────────
    log("Building multiscale pyramid...")
    t0 = time.perf_counter()
    pyramid = build_multiscale_pyramid(
        volume, args.num_levels,
        use_gpu=pyramid_gpu, ngpus=args.ngpus,
    )
    del volume
    pyramid_time = time.perf_counter() - t0
    log(f"  Pyramid time: {pyramid_time:.2f}s ({len(pyramid)} levels)")

    # ── Write OME-Zarr ────────────────────────────────────────────────
    log("Writing OME-Zarr...")
    t0 = time.perf_counter()
    write_ome_zarr(pyramid, zarr_path, args.chunk_size,
                   args.compression, args.pixel_size_nm,
                   io_workers=args.io_workers)
    write_time = time.perf_counter() - t0
    log(f"  Write time: {write_time:.2f}s")

    # ── Compute output size ───────────────────────────────────────────
    zarr_size = get_dir_size(zarr_path)

    # ── Create preview zarr (middle N slices for fast transfer) ───────
    import zarr as zarr_lib
    nz_full = pyramid[0].shape[0]
    n_preview = min(args.preview_slices, nz_full)
    # Extract from the MIDDLE of the volume for more representative data
    mid_start = (nz_full - n_preview) // 2
    preview_path = os.path.join(args.output_dir, "preview.zarr")
    log(f"Creating preview.zarr "
        f"({n_preview} of {nz_full} slices, z={mid_start}..{mid_start + n_preview})...")
    t0 = time.perf_counter()

    full_zarr = zarr_lib.open(zarr_path, mode="r")
    level0 = full_zarr["0"] if isinstance(full_zarr, zarr_lib.Group) and "0" in full_zarr else full_zarr
    _, ny_full, nx_full = level0.shape

    import numcodecs
    preview_store = zarr_lib.open_array(
        preview_path, mode="w", zarr_format=2,
        shape=(n_preview, ny_full, nx_full),
        chunks=(min(n_preview, args.chunk_size), args.chunk_size, args.chunk_size),
        dtype=level0.dtype,
        compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE),
    )
    # Copy middle slices in chunks to avoid loading everything at once
    chunk_z = args.chunk_size
    for z_start in range(0, n_preview, chunk_z):
        z_end = min(z_start + chunk_z, n_preview)
        src_start = mid_start + z_start
        src_end = mid_start + z_end
        preview_store[z_start:z_end] = level0[src_start:src_end]

    # Store the source offset so the viewer knows which region this covers
    preview_store.attrs["preview_z_start"] = int(mid_start)
    preview_store.attrs["preview_z_end"] = int(mid_start + n_preview)
    preview_store.attrs["full_z_size"] = int(nz_full)

    preview_time = time.perf_counter() - t0
    preview_size = get_dir_size(preview_path)
    log(f"  Preview: {n_preview} slices, {preview_size / (1024**2):.1f} MB, "
        f"{preview_time:.1f}s")

    # ── Tar full zarr (single file for efficient bulk transfer) ───────
    tar_path = None
    tar_size = 0
    tar_time = 0.0
    if not args.no_tar:
        import subprocess
        tar_filename = "reconstruction.ome.zarr.tar"
        tar_path = os.path.join(args.output_dir, tar_filename)
        log(f"Tarring full zarr → {tar_filename}...")
        t0 = time.perf_counter()
        subprocess.run(
            ["tar", "cf", tar_filename, "reconstruction.ome.zarr"],
            cwd=args.output_dir, check=True,
        )
        tar_time = time.perf_counter() - t0
        tar_size = os.path.getsize(tar_path)
        log(f"  Tar: {tar_size / (1024**3):.2f} GB, {tar_time:.1f}s")
    else:
        log("Skipping tar (--no-tar)")

    # ── Completion marker ─────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    done_info = {
        "status": "complete",
        "stage": "tiff_to_zarr",
        "zarr_path": zarr_path,
        "shape": list(pyramid[0].shape),
        "chunk_size": args.chunk_size,
        "num_levels": len(pyramid),
        "compression": args.compression,
        "zarr_size_bytes": zarr_size,
        "preview_path": preview_path,
        "preview_shape": [n_preview, ny_full, nx_full],
        "preview_size_bytes": preview_size,
        "tar_path": tar_path,
        "tar_size_bytes": tar_size,
        "total_time_s": round(total_time, 2),
        "load_time_s": round(load_time, 2),
        "pyramid_time_s": round(pyramid_time, 2),
        "write_time_s": round(write_time, 2),
        "preview_time_s": round(preview_time, 2),
        "tar_time_s": round(tar_time, 2),
        "ngpus": args.ngpus if args.use_gpu else 0,
        "io_workers": args.io_workers,
    }
    done_path = os.path.join(args.output_dir, "DONE_ZARR")
    with open(done_path, "w") as f:
        json.dump(done_info, f, indent=2)

    log(f"\nDone. Total: {total_time:.1f}s "
        f"(load: {load_time:.1f}s, pyramid: {pyramid_time:.1f}s, "
        f"write: {write_time:.1f}s, preview: {preview_time:.1f}s, "
        f"tar: {tar_time:.1f}s)")
    log(f"Zarr size: {zarr_size / (1024**2):.1f} MB")
    log(f"Preview: {n_preview} slices, {preview_size / (1024**2):.1f} MB")
    log(f"Tar: {tar_size / (1024**3):.2f} GB")
    log(f"Output: {zarr_path}")


if __name__ == "__main__":
    main()
