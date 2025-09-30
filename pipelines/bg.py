from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation, AutoImageProcessor

class BackgroundRemover:
    def __init__(
        self,
        model_id: str = "briaai/RMBG-1.4",
        input_size: int = 1024,
        threshold: float = 0.05,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id
        self.input_size = input_size
        self.threshold = threshold
        self.device = device
        self.dtype = dtype  # if None, we’ll pick a sensible default
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return

        # Choose dtype automatically when not specified
        if self.dtype is None:
            if self.device.startswith("cuda"):
                # Use fp16 on GPU if supported, else fall back to float32
                self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                self.dtype = torch.float32

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = (
            AutoModelForImageSegmentation
            .from_pretrained(self.model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        self.load()

        # Preprocess; many RMBG configs ignore size, but we pass it if supported.
        proc_kwargs = {"return_tensors": "pt"}
        # If the processor supports resizing via `size`, honor input_size.
        if hasattr(self.processor, "size") or "size" in getattr(self.processor, "init_kwargs", {}):
            proc_kwargs["size"] = {"height": self.input_size, "width": self.input_size}

        inputs = self.processor(images=image, **proc_kwargs)
        # Move tensor inputs to device/dtype
        inputs = {k: (v.to(self.device, dtype=self.dtype) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # RMBG-1.4 exposes a single-channel matte in logits
        # Shape: (B, 1, H', W')
        logits = outputs.logits
        matte = logits.sigmoid()

        # Upsample to original image size
        h, w = image.size[1], image.size[0]
        matte = F.interpolate(matte, size=(h, w), mode="bilinear", align_corners=False)
        matte = matte.squeeze().clamp(0, 1)

        # Threshold to make binary alpha (keep soft matte if you prefer)
        m = (matte > self.threshold).float()

        # Compose RGBA
        np_img = np.array(image, copy=False)
        if np_img.ndim == 2:  # grayscale → 3ch
            np_img = np.stack([np_img] * 3, axis=-1)

        alpha = (m.detach().cpu().numpy() * 255).astype(np.uint8)
        rgba = np.dstack([np_img, alpha])

        return Image.fromarray(rgba, mode="RGBA")
