
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation, AutoImageProcessor

class BackgroundRemover:
    def __init__(self, model_id: str = "mateenahmed/isnet-background-remover", input_size: int = 1024, threshold: float = 0.05, device="cuda"):
        self.model_id = model_id
        self.input_size = input_size
        self.threshold = threshold
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        if self.model is None:
            self.model = AutoModelForImageSegmentation.from_pretrained(self.model_id).to(self.device).eval()
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        self.load()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        mask = outputs.logits.sigmoid()  # (1,1,H,W)
        mask = torch.nn.functional.interpolate(mask, size=image.size[::-1], mode="bilinear", align_corners=False)
        mask = mask.squeeze().clamp(0,1)
        m = (mask > self.threshold).float()
        # Compose alpha image
        np_img = np.array(image).astype(np.uint8)
        if np_img.ndim == 2:
            np_img = np.stack([np_img]*3, axis=-1)
        alpha = (m.cpu().numpy()*255).astype(np.uint8)
        rgba = np.dstack([np_img, alpha])
        return Image.fromarray(rgba, mode="RGBA")
