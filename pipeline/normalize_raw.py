#!/usr/bin/env python
"""
Normalize a raw TXM HDF5 file (SSRL 6-2c exchange format) on Perlmutter.

Reads raw exchange HDF5 with exchange/data, exchange/data_dark,
exchange/data_white, exchange/theta.  Applies flat-field normalization
and writes output in the format expected by reconstruct_gpu.py.

GPU acceleration via CuPy when available (A100: ~10x faster compute).
Falls back to NumPy CPU if CuPy is not installed.

Output:
    normalized_projections.hdf5  — dataset 'process/normalized/data'
    import_metadata.json         — angles, shape, pixel size
    DONE_NORMALIZE               — JSON marker with status and timing

Usage:
    python normalize_raw.py --raw-h5 /path/to/raw.h5 --output-dir /path/to/output/
    python normalize_raw.py --raw-h5 /path/to/raw.h5 --output-dir /path/to/output/ --no-gpu
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

# Try CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _normalize_chunk_gpu(proj_chunk_np, dark_mean_gpu, denom_gpu):
    """Normalize a chunk of projections on GPU. Returns NumPy array."""
    chunk_gpu = cp.asarray(proj_chunk_np, dtype=cp.float32)
    chunk_gpu -= dark_mean_gpu
    chunk_gpu /= denom_gpu
    cp.clip(chunk_gpu, 0.0, 2.0, out=chunk_gpu)
    result = cp.asnumpy(chunk_gpu)
    del chunk_gpu
    return result


def _normalize_chunk_cpu(proj_chunk_np, dark_mean, denom):
    """Normalize a chunk of projections on CPU. Returns NumPy array."""
    chunk = proj_chunk_np.astype(np.float32)
    chunk -= dark_mean
    chunk /= denom
    np.clip(chunk, 0.0, 2.0, out=chunk)
    return chunk


def normalize(raw_h5_path, output_dir, use_gpu=True):
    """Flat-field normalize raw projections and save in pipeline format.

    Uses triple-buffered pipeline: read chunk N+1 while GPU normalizes
    chunk N and writes chunk N-1 to disk, overlapping I/O and compute.
    """
    os.makedirs(output_dir, exist_ok=True)

    use_gpu = use_gpu and HAS_CUPY
    backend = "GPU (CuPy)" if use_gpu else "CPU (NumPy)"

    print(f"Normalizing: {raw_h5_path}", flush=True)
    print(f"Output dir:  {output_dir}", flush=True)
    print(f"Backend:     {backend}", flush=True)

    if use_gpu:
        dev = cp.cuda.Device(0)
        print(f"GPU:         {dev.id} — {dev.mem_info[1] / 1e9:.1f} GB total", flush=True)

    t0 = time.time()

    with h5py.File(raw_h5_path, "r") as f:
        print("  Loading projections...", flush=True)
        proj = f["exchange/data"][:]         # (nangles, nrows, ncols) uint16
        print(f"    proj shape: {proj.shape}, dtype: {proj.dtype}", flush=True)

        print("  Loading dark fields...", flush=True)
        dark = f["exchange/data_dark"][:]    # (ndark, nrows, ncols) uint16
        print(f"    dark shape: {dark.shape}", flush=True)

        print("  Loading flat fields...", flush=True)
        flat = f["exchange/data_white"][:]   # (nflat, nrows, ncols) uint16
        print(f"    flat shape: {flat.shape}", flush=True)

        print("  Loading theta...", flush=True)
        theta_deg = f["exchange/theta"][:]   # (nangles,) float32, in degrees
        print(f"    theta: {theta_deg.shape}, range [{theta_deg.min():.1f}, {theta_deg.max():.1f}] deg", flush=True)

        # Extract metadata
        pixel_size = 0.0
        energy = 0.0
        try:
            pixel_size = float(f["measurement/instrument/detector/pixel_size"][0])
        except Exception:
            pass
        try:
            energy = float(f["measurement/instrument/monochromator/energy"][0])
        except Exception:
            pass

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", flush=True)

    # Flat-field correction: normalized = (proj - dark) / (flat - dark)
    print("  Computing flat-field correction...", flush=True)
    t1 = time.time()

    dark_mean = np.mean(dark, axis=0, dtype=np.float32)  # (nrows, ncols)
    flat_mean = np.mean(flat, axis=0, dtype=np.float32)   # (nrows, ncols)
    del dark, flat  # Free memory

    # Avoid division by zero
    denom = flat_mean - dark_mean
    denom[denom < 1.0] = 1.0

    # Upload correction frames to GPU once (they stay resident)
    if use_gpu:
        dark_mean_gpu = cp.asarray(dark_mean)
        denom_gpu = cp.asarray(denom)

    nangles = proj.shape[0]
    nrows, ncols = proj.shape[1], proj.shape[2]

    # Chunk size: balance GPU utilization vs memory.
    # 200 projections × 2048 × 2048 × 4 bytes ≈ 3.2 GB on GPU — safe for A100 40 GB.
    chunk_size = 200 if use_gpu else 100

    # Convert theta to radians
    theta_rad = np.deg2rad(theta_deg).astype(np.float64).tolist()

    # Create output HDF5 with pre-allocated dataset (no compression for speed)
    out_h5 = os.path.join(output_dir, "normalized_projections.hdf5")
    print(f"  Saving to {out_h5} ({backend}, chunk_size={chunk_size})...", flush=True)

    with h5py.File(out_h5, "w") as fout:
        ds = fout.create_dataset(
            "process/normalized/data",
            shape=(nangles, nrows, ncols),
            dtype=np.float32,
            chunks=(1, nrows, ncols),
        )

        write_future = None

        def _write_chunk(data, start, end):
            ds[start:end] = data

        with ThreadPoolExecutor(max_workers=1) as writer:
            for start in range(0, nangles, chunk_size):
                end = min(start + chunk_size, nangles)

                # Normalize this chunk
                if use_gpu:
                    normalized = _normalize_chunk_gpu(
                        proj[start:end], dark_mean_gpu, denom_gpu
                    )
                else:
                    normalized = _normalize_chunk_cpu(
                        proj[start:end], dark_mean, denom
                    )

                # Wait for previous write to finish before submitting next
                if write_future is not None:
                    write_future.result()

                # Submit async write (runs in background thread)
                write_future = writer.submit(_write_chunk, normalized, start, end)

                if (end % 200 == 0) or end == nangles:
                    elapsed = time.time() - t1
                    rate = end / elapsed if elapsed > 0 else 0
                    print(f"    Normalized {end}/{nangles} projections "
                          f"({rate:.0f} proj/s)", flush=True)

            # Wait for final write
            if write_future is not None:
                write_future.result()

    # Free memory
    del proj
    if use_gpu:
        del dark_mean_gpu, denom_gpu
        cp.get_default_memory_pool().free_all_blocks()

    norm_time = time.time() - t1
    out_size_gb = os.path.getsize(out_h5) / 1e9
    print(f"  Normalization done in {norm_time:.1f}s", flush=True)
    print(f"  Normalized HDF5: {out_h5} ({out_size_gb:.1f} GB)", flush=True)

    # Save metadata
    meta = {
        "metadata_type": "SSRL62C_Normalized",
        "angles_rad": theta_rad,
        "pixel_size": pixel_size,
        "energy_float": energy,
        "num_projections": int(nangles),
        "image_height": int(nrows),
        "image_width": int(ncols),
        "source_file": os.path.basename(raw_h5_path),
    }
    meta_path = os.path.join(output_dir, "import_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}", flush=True)

    total = time.time() - t0

    # Write DONE marker
    done_marker = {
        "status": "success",
        "backend": backend,
        "total_time_s": round(total, 1),
        "load_time_s": round(load_time, 1),
        "normalize_time_s": round(norm_time, 1),
        "output_h5": out_h5,
        "output_size_gb": round(out_size_gb, 1),
        "shape": [int(nangles), int(nrows), int(ncols)],
    }
    done_path = os.path.join(output_dir, "DONE_NORMALIZE")
    with open(done_path, "w") as f:
        json.dump(done_marker, f, indent=2)
    print(f"  Wrote {done_path}", flush=True)

    print(f"\nDone! Total time: {total:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize raw TXM HDF5 on Perlmutter")
    parser.add_argument("--raw-h5", required=True,
                        help="Path to raw exchange HDF5 scan file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for normalized output")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only normalization (skip CuPy)")
    args = parser.parse_args()
    normalize(args.raw_h5, args.output_dir, use_gpu=not args.no_gpu)
