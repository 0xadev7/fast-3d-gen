import os, torch
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download

T2I_MODEL = os.environ.get("T2I_MODEL", "black-forest-labs/FLUX.1-schnell")
# prefer a Transformers-native model here:
BG_MODEL  = os.environ.get("BG_MODEL", "briaai/RMBG-1.4")
TRELLIS_MODEL = os.environ.get("TRELLIS_IMAGE_MODEL", "JeffreyXiang/TRELLIS-image-large")

print(f"Warmdownloading: {T2I_MODEL}, {BG_MODEL}, {TRELLIS_MODEL}")

# T2I (Flux)
try:
    _ = FluxPipeline.from_pretrained(T2I_MODEL, torch_dtype=torch.float16)
    print("✓ FLUX downloaded")
except Exception as e:
    print("! FLUX download skipped/error:", e)

# Background remover
try:
    from transformers import AutoImageProcessor, AutoModelForImageSegmentation

    try:
        # First, try standard load (works for RMBG)
        _ = AutoImageProcessor.from_pretrained(BG_MODEL)
        _ = AutoModelForImageSegmentation.from_pretrained(BG_MODEL)
        print("✓ BG model downloaded")
    except Exception:
        # Fallback: allow custom modeling code (if repo provides it)
        _ = AutoImageProcessor.from_pretrained(BG_MODEL, trust_remote_code=True)
        _ = AutoModelForImageSegmentation.from_pretrained(BG_MODEL, trust_remote_code=True)
        print("✓ BG model (remote code) downloaded")
except Exception as e:
    print("! BG model download skipped/error:", e)

# TRELLIS snapshot
try:
    snapshot_download(repo_id=TRELLIS_MODEL, allow_patterns=["*"])
    print("✓ TRELLIS weights cached")
except Exception as e:
    print("! Trellis snapshot skipped/error:", e)

print("Warm download complete.")
