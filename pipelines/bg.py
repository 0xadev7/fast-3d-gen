from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForImageSegmentation,
    AutoImageProcessor,
    AutoProcessor,
)

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def _resize_with_pad(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    """Keep aspect ratio, letterbox to size×size."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(img_resized, (left, top))
    return canvas

def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[..., None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[..., None, None]
    return (x - mean) / std

class BackgroundRemover:
    def __init__(
        self,
        model_id: str = "briaai/RMBG-1.4",
        input_size: int = 1024,
        threshold: Optional[float] = 0.05,   # set to None for soft alpha
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        self.input_size = input_size
        self.threshold = threshold
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._used_manual_preproc = False

    def load(self):
        if self.model is not None:
            return

        # dtype defaulting
        if self.dtype is None:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32

        # 1) Try standard image processor
        proc = None
        try:
            proc = AutoImageProcessor.from_pretrained(self.model_id)
        except Exception:
            # 2) Try AutoProcessor with trust_remote_code
            try:
                proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception:
                proc = None

        self.processor = proc
        self.model = (
            AutoModelForImageSegmentation
            .from_pretrained(self.model_id, torch_dtype=self.dtype, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

        # If neither processor variant loaded, we'll manual-preprocess
        self._used_manual_preproc = self.processor is None

    def _manual_preprocess(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback when the repo doesn't provide a usable processor."""
        img = _to_rgb(image)
        img_sq = _resize_with_pad(img, self.input_size)  # size x size
        arr = np.array(img_sq).astype(np.float32) / 255.0  # HWC, [0,1]
        # to CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 3xHxW
        tensor = tensor.to(self.device, dtype=self.dtype)
        tensor = _imagenet_norm(tensor)
        tensor = tensor.unsqueeze(0)  # 1x3xHxW
        return {"pixel_values": tensor, "pad_applied": True}

    def _processor_preprocess(self, image: Image.Image) -> Dict[str, Any]:
        # Respect input_size if the processor accepts a 'size' kwarg
        kwargs = {"return_tensors": "pt"}
        try:
            kwargs["size"] = {"height": self.input_size, "width": self.input_size}
        except Exception:
            pass  # some processors don’t accept size

        inputs = self.processor(images=image, **kwargs)
        # move tensors to device/dtype
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device, dtype=self.dtype)
            else:
                prepared[k] = v
        prepared["pad_applied"] = False
        return prepared

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        self.load()

        if self.processor is None:
            batch = self._manual_preprocess(image)
        else:
            batch = self._processor_preprocess(image)

        outputs = self.model(**{k: v for k, v in batch.items() if k not in ("pad_applied",)})
        # Try common fields for segmentation/matting heads
        logits = getattr(outputs, "logits", None)
        if logits is None:
            # Some remote-code models return 'predictions' or 'matte'
            logits = getattr(outputs, "predictions", None)
        if logits is None:
            logits = getattr(outputs, "matte", None)
        if logits is None:
            raise RuntimeError("RMBG model did not return a recognizable logits/matte tensor.")

        # Expect Bx1xH’xW’
        matte = torch.sigmoid(logits)

        # If we letterboxed in manual preprocess, we should unpad after upsampling.
        orig_w, orig_h = image.size
        matte_up = F.interpolate(matte, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        matte_up = matte_up.squeeze().clamp(0, 1)  # HxW

        # If you prefer soft alpha, set threshold=None
        if self.threshold is None:
            alpha_t = (matte_up * 255.0).to(torch.uint8)
        else:
            alpha_t = ((matte_up > self.threshold).float() * 255.0).to(torch.uint8)

        np_img = np.array(image)
        if np_img.ndim == 2:  # grayscale
            np_img = np.stack([np_img] * 3, axis=-1)

        alpha = alpha_t.detach().cpu().numpy()
        rgba = np.dstack([np_img, alpha])
        return Image.fromarray(rgba, mode="RGBA")
