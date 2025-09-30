
from __future__ import annotations
import io, os, math, time, random
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from PIL import Image

def seed_everything(seed: Optional[int] = None):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def to_pil(image):
    if isinstance(image, Image.Image): return image
    if hasattr(image, "to_pil_image"):
        return image.to_pil_image()
    try:
        from PIL import Image as _Image
        return _Image.fromarray(image)
    except Exception:
        raise

def save_bytesio(pathless_bytes: bytes) -> io.BytesIO:
    bio = io.BytesIO()
    bio.write(pathless_bytes)
    bio.seek(0)
    return bio

def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w,h)
    left = (w - s)//2
    top  = (h - s)//2
    return img.crop((left, top, left+s, top+s))

class Timer:
    def __init__(self): self.stamps = {}
    def stamp(self, k): self.stamps[k] = time.time()
    def since(self, k): return time.time() - self.stamps.get(k, time.time())
