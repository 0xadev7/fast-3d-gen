
from __future__ import annotations
from typing import List
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from aesthetics_predictor import AestheticsPredictorV1

class SelfValidator:
    def __init__(self, clip_model: str = "ViT-B/32", min_clip_score: float = 0.24, min_aesthetic: float = 4.5, device="cuda"):
        import clip  # from clip-anytorch
        self.device = device
        self.clip_model, self.clip_preproc = clip.load(clip_model, device=device, download_root=None)
        self.clip_model.eval()
        self.min_clip = min_clip_score
        self.min_aes = min_aesthetic
        self.aesthetic = AestheticsPredictorV1(device=device)

    @torch.inference_mode()
    def score_clip(self, prompt: str, images: List[Image.Image]) -> float:
        import clip
        text = clip.tokenize([prompt]).to(self.device)
        img_tensors = [self.clip_preproc(im).unsqueeze(0).to(self.device) for im in images]
        imgs = torch.cat(img_tensors, dim=0)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            img_features  = self.clip_model.encode_image(imgs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            img_features  = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ text_features.T).squeeze()
        return float(sims.mean().item())

    @torch.inference_mode()
    def score_aesthetic(self, images: List[Image.Image]) -> float:
        scores = [float(self.aesthetic(im)) for im in images]
        return float(np.mean(scores))

    def validate(self, prompt: str, images: List[Image.Image]) -> bool:
        clip_s = self.score_clip(prompt, images)
        aes_s  = self.score_aesthetic(images)
        return (clip_s >= self.min_clip) and (aes_s >= self.min_aes), (clip_s, aes_s)
