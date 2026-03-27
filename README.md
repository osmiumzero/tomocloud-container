# TomoCloud Pipeline Container

GPU-accelerated tomographic reconstruction and AI segmentation container for DOE facility pipelines.

## Stack

- **tomocupy** -- GPU reconstruction (built from source, CUDA 12.2)
- **SAM 2.1 + SAM3** -- AI segmentation (Facebook Research)
- **PyTorch** -- cu121 (CUDA 12.1 runtime)
- **CuPy, Zarr, OME-Zarr** -- GPU array processing + cloud-native storage
- **Globus SDK** -- data transfer integration
- Python 3.11, numpy < 2.0

## Repository Structure

```
tomocloud-container/
  apptainer/              # Apptainer/Singularity container
    txm-pipeline.def      # Definition file
    build.sh              # Build script (fakeroot, no root needed)

  docker/                 # OCI/Docker container
    Dockerfile            # Multi-stage build (builder + runtime)
    entrypoint.sh         # Container entrypoint
    .dockerignore

  deploy/                 # Deployment and registry scripts
    push-oci.sh           # Push SIF to OCI registry (ORAS)
    build-and-push-docker.sh  # Build Docker image + push to registry
    run-with-globus.sh    # Run with Globus/HuggingFace bind mounts
```

## Quick Start

### Option 1: Apptainer (HPC / Slurm)

```bash
cd apptainer/
bash build.sh
# Output: txm-pipeline.sif (~6-7 GB)
```

### Option 2: Docker / OCI (AWS, Kubernetes, Cloud)

```bash
cd docker/
docker build -t tomocloud-pipeline:latest .
```

### Push to Registry

From Apptainer SIF:
```bash
bash deploy/push-oci.sh txm-pipeline.sif ghcr.io/<org>/tomocloud-pipeline:latest
```

From Docker build:
```bash
bash deploy/build-and-push-docker.sh ghcr.io/<org>/tomocloud-pipeline:latest
```

## Running

### Apptainer + Slurm

```bash
srun --gpus=4 --time=03:00:00 \
  apptainer exec --nv \
    --bind /data:/data \
    --bind /models:/models \
    --bind ~/.cache/huggingface:/hf_cache \
    txm-pipeline.sif python /app/reconstruct_gpu.py ...
```

### Docker / AWS

```bash
docker run --gpus all \
  -v /data:/data \
  -v /models:/models \
  -v ~/.cache/huggingface:/hf_cache \
  tomocloud-pipeline:latest python /app/reconstruct_gpu.py ...
```

### With Globus Integration

```bash
bash deploy/run-with-globus.sh apptainer \
  --sif txm-pipeline.sif \
  --data-dir /path/to/data \
  --output-dir /path/to/output \
  --model-dir /path/to/models \
  -- python /app/reconstruct_gpu.py ...
```

## CUDA Compatibility

- Base image: `nvidia/cuda:12.2.2` (devel for build, runtime for final)
- Tested drivers: 535.x+ (S3DF L40S), 550.x+ (AWS)
- PyTorch cu121 wheels are forward-compatible with CUDA 12.2+ drivers

## Build Notes

- **tomocupy MUST be built with `--no-build-isolation`** -- pip's isolated build env produces .so files with no embedded CUDA device code
- **numpy must stay < 2.0** -- tomocupy and scipy 1.13 are compiled against numpy 1.x ABI
- **SAM3 requires HuggingFace token** for gated model access -- bind mount `~/.cache/huggingface` into the container

## Tested Platforms

| Platform | GPUs | Status |
|----------|------|--------|
| SLAC S3DF (ada) | L40S 46GB | Validated |
| ALCF Polaris | A100 40GB | Validated |
| NERSC Perlmutter | A100 40GB | Validated |
| AWS (via IRI API) | TBD | In progress |

## Facilities

This container is designed for use across D4 tomography pipelines:
- **SSRL** (SLAC) -- beamline 6-2 TXM
- **APS** (ANL) -- tomography beamlines
- **ALS** (LBNL) -- tomography beamlines

Data transfer via Globus enables seamless integration with any facility endpoint.
