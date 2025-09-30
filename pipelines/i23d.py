
from __future__ import annotations
import os, tempfile, math
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import imageio
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

class ImageTo3D:
    def __init__(self, model_id: str, steps_sparse: int = 8, steps_slat: int = 8, device="cuda", small_model=False):
        self.model_id = model_id
        self.steps_sparse = steps_sparse
        self.steps_slat = steps_slat
        self.device = device
        self.small_model = small_model
        self.pipe = None

    def load(self):
        if self.pipe is None:
            mid = self.model_id
            if self.small_model:
                mid = "JeffreyXiang/TRELLIS-image-base"
            self.pipe = TrellisImageTo3DPipeline.from_pretrained(mid)
            self.pipe.cuda()

    @torch.inference_mode()
    def infer(self, image: Image.Image):
        self.load()
        # Trellis accepts PIL.Image; tune steps via sampler params
        outputs = self.pipe.run(
            image,
            sparse_structure_sampler_params={ "steps": self.steps_sparse, "cfg_strength": 7.0 },
            slat_sampler_params={ "steps": self.steps_slat, "cfg_strength": 3.0 },
        )
        return outputs

    def export_ply(self, outputs, path: str):
        # outputs['gaussian'] is a list; take first
        g = outputs['gaussian'][0]
        g.save_ply(path)

    def export_glb(self, outputs, path: str):
        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.9, texture_size=1024)
        glb.export(path)

    def render_turntable(self, outputs, video_path: str, seconds: int = 5, fps: int = 24, yaw_degrees: int = 360, elevation_degrees: int = 15):
        g = outputs['gaussian'][0]
        frames = render_utils.render_video(
            g,
            yaw_range=(0, yaw_degrees),
            elevation=elevation_degrees,
            num_frames=seconds*fps,
        )['color']
        imageio.mimsave(video_path, frames, fps=fps)
