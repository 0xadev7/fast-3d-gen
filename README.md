
# Fast 3D Miner — Generation Service (Competitive)

An **optimized replacement** for the `generation` service in the Three‑Gen Subnet.  
It keeps the *same HTTP contract* (`/generate` and `/generate_video`) but swaps in a **faster, higher‑quality, commercially usable** pipeline:

- **Text → Image:** `FLUX.1‑schnell` (Apache‑2.0) via 🤗 Diffusers
- **Background Removal:** **ISNet** (MIT). More robust than `rembg` and commercially friendly
- **Image → 3D:** **TRELLIS image‑to‑3D** (MIT), exporting **Gaussian splats (PLY)** and **GLB**. Tuned for speed
- **Self‑Validation:** CLIP (MIT) + LAION Aesthetics (MIT); returns **204 No Content** on fail (for miner cooldown protection)
- **Video:** fast turntable render straight from the generated Gaussians

> ⚠️ This repo only replaces **/generation**. You can continue to use the upstream **miner neuron** without changes, pointing it to this service.


## Endpoints (compatible with `three-gen-subnet/generation`)

- `POST /generate/` — form encoded
  - **Request:** `prompt=<text>`
  - **Response:** binary **PLY** stream (Gaussian splats). Status **204** if validation fails

- `POST /generate_video/` — form encoded
  - **Request:** `prompt=<text>`
  - **Response:** binary **MP4** stream (turntable). Status **204** if validation fails

Example:

```bash
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
```

## Why this is faster & better

- **FLUX.1‑schnell** generates high‑quality images in **1–4 steps** (we use 2–3 by default)
- **ISNet** gives cleaner alpha mattes than `rembg` and is MIT‑licensed
- **TRELLIS** is a **feed‑forward** image‑to‑3D model (no SDS optimization loop), supports **Gaussian splats** and **meshes**, and exports **PLY** quickly
- **Asynchronous FastAPI** with CUDA‑warm start and optional pre‑downloaded weights
- **Self‑validation** blocks low‑quality outputs and automatically retries once

## Quick start (local)

```bash
# 1) Python 3.10+ and CUDA 12.x recommended
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# (Optional) pre-download weights to populate the HF cache and avoid cold starts
python scripts/download_weights.py

# 2) Run the service
python app.py --host 0.0.0.0 --port 8093
# or PM2 (see pm2/generation.config.js)
```

## RunPod deployment

### Pods (recommended for steady mining)
- 1× **RTX 4090** (24 GB) or **L4** (24 GB)
- Build once, **keep models warm**; best cost for continuous throughput
- See `docker/Dockerfile` and `docker/runpod-start.sh`

### Serverless (good for bursty load)
- Pay‑per‑second; **cold starts** mitigated by `scripts/download_weights.py` and warm‑pool config
- For consistent 30 s SLAs the **pod** route is still more predictable

## Configuration

Edit `configs/service.yaml`:

- **t2i:** number of steps, resolution, guidance scale
- **trellis:** steps, quality/speed mode
- **validation:** thresholds, max retries
- **render:** turntable fps, seconds

## Files

- `app.py` — FastAPI server + endpoints
- `pipelines/` — T2I, background removal, 3D, validation, utils
- `pm2/generation.config.js` — mirrors the upstream process manager
- `docker/` — Dockerfile + RunPod start script
- `licenses/` — summary of licenses and links

## Notes

- On first run, libraries like TRELLIS may compile backends (`flash‑attn`, `spconv`, etc.). Do this once in your image or persistent volume
- If TRELLIS OOMs on 24 GB, set `trellis.small_model: true` in `configs/service.yaml`
- If validation fails twice, the service **returns 204** as per subnet guidance

