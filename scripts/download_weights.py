
import os, json, sys, torch

# Minimal warm downloads to avoid serverless cold starts.
# Respect HF cache and do not force redownloads.
from diffusers import FluxPipeline
from transformers import AutoModelForImageSegmentation, AutoProcessor
from huggingface_hub import snapshot_download

T2I_MODEL = os.environ.get("T2I_MODEL", "black-forest-labs/FLUX.1-schnell")
BG_MODEL  = os.environ.get("BG_MODEL", "mateenahmed/isnet-background-remover")
TRELLIS_MODEL = os.environ.get("TRELLIS_IMAGE_MODEL", "JeffreyXiang/TRELLIS-image-large")

print(f"Warmdownloading: {T2I_MODEL}, {BG_MODEL}, {TRELLIS_MODEL}")

# T2I
try:
    pipe = FluxPipeline.from_pretrained(T2I_MODEL, torch_dtype=torch.float16, variant="fp16")
    del pipe
    print("✓ FLUX downloaded")
except Exception as e:
    print("! FLUX download skipped/error:", e)

# Background model (weights via transformers)
try:
    from transformers import AutoModelForImageSegmentation
    _ = AutoModelForImageSegmentation.from_pretrained(BG_MODEL)
    print("✓ ISNet downloaded")
except Exception as e:
    print("! ISNet download skipped/error:", e)

# TRELLIS (relies on HF snapshot)
try:
    snapshot_download(repo_id=TRELLIS_MODEL, allow_patterns=["*"])
    print("✓ TRELLIS weights cached")
except Exception as e:
    print("! Trellis snapshot skipped/error:", e)

print("Warm download complete.")
