#!/usr/bin/env python
"""
GPU reconstruction using tomocupy for NERSC Perlmutter.

Reads normalized projections from the pipeline's normalize step,
adds exchange-format datasets to the HDF5 (via hard links, no data
copy), and runs tomocupy for GPU-accelerated reconstruction on all
available GPUs (4x A100 per Perlmutter node).

Usage:
    python reconstruct_gpu.py --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def log(msg):
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GPU TomocuPy reconstruction")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h5-file", default="normalized_projections.hdf5")
    parser.add_argument("--meta-file", default="import_metadata.json")
    parser.add_argument("--algorithm", default="fourierrec",
                        choices=["fourierrec", "lprec", "linerec"])
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16"])
    parser.add_argument("--nsino-per-chunk", type=int, default=8,
                        help="Sinograms per GPU chunk (tune for memory)")
    parser.add_argument("--nproj-per-chunk", type=int, default=8,
                        help="Projections per GPU chunk")
    parser.add_argument("--fbp-filter", default="parzen",
                        choices=["none", "ramp", "shepp", "hann", "hamming",
                                 "parzen", "cosine", "cosine2"])
    parser.add_argument("--rotation-axis", type=float, default=-1.0,
                        help="Rotation axis position (-1 for auto)")
    parser.add_argument("--remove-stripe-method", default="none",
                        choices=["none", "fw", "ti", "vo-all"],
                        help="Stripe removal method (none|fw|ti|vo-all)")
    parser.add_argument("--save-format", default="tiff",
                        choices=["tiff", "h5"])
    return parser.parse_args()


def add_exchange_datasets(h5_path, meta_path):
    """Add exchange-format datasets to the normalized HDF5 via hard links.

    tomocupy expects /exchange/data, /exchange/data_dark,
    /exchange/data_white, and /exchange/theta.  Since our data is
    already flat-field corrected (I/I0), dark=0 and flat=1 make
    tomocupy's normalization an identity operation.  The hard link
    for /exchange/data avoids copying the projection data.
    """
    import h5py

    with open(meta_path, "r") as f:
        meta = json.load(f)
    theta_deg = np.degrees(np.array(meta["angles_rad"], dtype=np.float64))

    with h5py.File(h5_path, "a") as f:
        if "exchange" in f:
            del f["exchange"]

        shape = f["process/normalized/data"].shape
        # Hard link — instant, no data copy
        f["/exchange/data"] = f["process/normalized/data"]
        f.create_dataset("/exchange/data_dark",
                         data=np.zeros((1, shape[1], shape[2]),
                                       dtype=np.float32))
        f.create_dataset("/exchange/data_white",
                         data=np.ones((1, shape[1], shape[2]),
                                      dtype=np.float32))
        f.create_dataset("/exchange/theta", data=theta_deg)

    return shape, meta, theta_deg


def remove_exchange_datasets(h5_path):
    """Remove the temporary exchange datasets from the HDF5 file."""
    import h5py
    try:
        with h5py.File(h5_path, "a") as f:
            if "exchange" in f:
                del f["exchange"]
    except Exception as e:
        log(f"  Warning: could not clean up exchange datasets: {e}")


def main():
    args = parse_args()

    h5_path = os.path.join(args.data_dir, args.h5_file)
    meta_path = os.path.join(args.data_dir, args.meta_file)
    tiff_dir = os.path.join(args.output_dir, "tiff_slices")

    log(f"GPU reconstruction (tomocupy): algorithm={args.algorithm}, "
        f"dtype={args.dtype}")
    t_total = time.perf_counter()

    # ── Add exchange datasets to normalized HDF5 (no data copy) ──────
    log(f"Preparing exchange-format datasets in {h5_path} ...")
    t0 = time.perf_counter()
    shape, meta, theta_deg = add_exchange_datasets(h5_path, meta_path)
    log(f"  Data shape: {shape}, exchange datasets added in "
        f"{time.perf_counter() - t0:.1f}s")

    # ── Run tomocupy reconstruction ──────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "tomocupy", "recon",
        "--file-name", h5_path,
        "--out-path-name", args.output_dir,
        "--reconstruction-type", "full",
        "--reconstruction-algorithm", args.algorithm,
        "--dtype", args.dtype,
        "--fbp-filter", args.fbp_filter,
        "--nsino-per-chunk", str(args.nsino_per_chunk),
        "--nproj-per-chunk", str(args.nproj_per_chunk),
        "--save-format", args.save_format,
        "--minus-log", "True",
    ]

    # Rotation axis: auto or manual
    if args.rotation_axis < 0:
        cmd.extend(["--rotation-axis-auto", "auto"])
    else:
        cmd.extend([
            "--rotation-axis-auto", "manual",
            "--rotation-axis", str(args.rotation_axis),
        ])

    # Stripe removal
    if args.remove_stripe_method != "none":
        cmd.extend(["--remove-stripe-method", args.remove_stripe_method])

    log(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, text=True, timeout=3600)
    recon_time = time.perf_counter() - t0
    log(f"  tomocupy finished in {recon_time:.1f}s "
        f"(returncode={result.returncode})")

    # Clean up exchange datasets regardless of outcome
    remove_exchange_datasets(h5_path)

    if result.returncode != 0:
        log("ERROR: tomocupy reconstruction failed")
        sys.exit(result.returncode)

    # ── Post-process: move output to expected structure ───────────────
    # tomocupy v1.1.0 outputs to <out_path_name>/recon/ (not <stem>_rec/)
    # Check both patterns for compatibility
    rec_dir = None
    for pattern in ["recon", "*_rec"]:
        matches = sorted(glob.glob(os.path.join(args.output_dir, pattern)))
        if matches and os.path.isdir(matches[0]):
            rec_dir = matches[0]
            break
    if rec_dir is None:
        rec_dir = args.output_dir
    log(f"  Reconstruction output dir: {rec_dir}")

    # Collect TIFFs from tomocupy output
    tiffs = sorted(
        glob.glob(os.path.join(rec_dir, "*.tiff"))
        + glob.glob(os.path.join(rec_dir, "*.tif"))
    )

    # Move to tiff_slices/ with pipeline-expected naming
    os.makedirs(tiff_dir, exist_ok=True)
    log(f"  Moving {len(tiffs)} TIFF slices to {tiff_dir} ...")
    for i, src in enumerate(tiffs):
        shutil.move(src, os.path.join(tiff_dir, f"slice_{i:04d}.tif"))

    # Remove tomocupy's output directory if it's separate
    if rec_dir != args.output_dir and os.path.isdir(rec_dir):
        shutil.rmtree(rec_dir)

    n_slices = len(tiffs)

    if n_slices > 0:
        import tifffile
        sample = tifffile.imread(os.path.join(tiff_dir, "slice_0000.tif"))
        recon_shape = (n_slices, sample.shape[0], sample.shape[1])
    else:
        recon_shape = (0, 0, 0)

    # ── Completion marker ────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    done_info = {
        "status": "complete",
        "shape": list(recon_shape),
        "slices": n_slices,
        "algorithm": args.algorithm,
        "gpu_mode": "tomocupy (auto multi-GPU)",
        "dtype": args.dtype,
        "recon_time_s": round(recon_time, 2),
        "total_time_s": round(total_time, 2),
    }
    with open(os.path.join(args.output_dir, "DONE"), "w") as f:
        json.dump(done_info, f, indent=2)

    log(f"\nDone. Total: {total_time:.1f}s (recon: {recon_time:.1f}s)")
    log(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
