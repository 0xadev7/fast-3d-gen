
from __future__ import annotations
import torch
from diffusers import FluxPipeline
from PIL import Image
from .utils import seed_everything

class T2IPipeline:
    def __init__(self, model_id: str, device: str = "cuda", steps: int = 3, guidance: float = 0.0, height: int = 768, width: int = 768, seed=None):
        self.model_id = model_id
        self.device = device
        self.steps = steps
        self.guidance = guidance
        self.height = height
        self.width = width
        self.seed = seed
        self.pipe = None

    def load(self):
        if self.pipe is None:
            self.pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
            self.pipe.to(self.device)
            self.pipe.vae.enable_tiling()  # lower VRAM spikes

    @torch.inference_mode()
    def __call__(self, prompt: str) -> Image.Image:
        seed = seed_everything(self.seed)
        self.load()
        out = self.pipe(
            prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.height,
            width=self.width,
        )
        return out.images[0]
