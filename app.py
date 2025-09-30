
from __future__ import annotations
import io, os, argparse, asyncio, tempfile
from typing import Optional
from fastapi import FastAPI, Form, Response, HTTPException
from loguru import logger
import yaml
from PIL import Image
import torch

from pipelines.t2i import T2IPipeline
from pipelines.bg import BackgroundRemover
from pipelines.i23d import ImageTo3D
from pipelines.validate import SelfValidator
from pipelines.utils import Timer, center_crop_to_square

app = FastAPI(title="Fast 3D Miner Generation Service", version="0.1.0")

class ServiceState:
    def __init__(self, cfg):
        device = f"cuda:{cfg['server'].get('cuda_device', 0)}" if torch.cuda.is_available() else "cpu"
        self.t2i = T2IPipeline(**cfg["t2i"], device=device)
        self.bg  = BackgroundRemover(**cfg["background"], device=device)
        self.i23d = ImageTo3D(**cfg["trellis"], device=device)
        self.val = SelfValidator(**cfg["validation"], device=device)
        self.cfg = cfg
        logger.info(f"Using device: {device}")

    def render_views(self, outputs) -> list[Image.Image]:
        # Render a few angles from Gaussians for validation
        from trellis.utils import render_utils
        g = outputs['gaussian'][0]
        frames = render_utils.render_video(g, yaw_range=(0,240), elevation=15, num_frames=3)['color']
        return [Image.fromarray(fr) for fr in frames]

state: Optional[ServiceState] = None

@app.on_event("startup")
async def _startup():
    global state
    cfg_path = os.environ.get("SERVICE_CONFIG", "configs/service.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    state = ServiceState(cfg)
    # Lazy-load heavy models after startup to keep boot snappy
    await asyncio.to_thread(state.t2i.load)
    await asyncio.to_thread(state.bg.load)
    await asyncio.to_thread(state.i23d.load)
    logger.info("Pipelines warmed.")

@app.post("/generate/")
async def generate(prompt: str = Form(...)):
    timer = Timer(); timer.stamp("t0")
    try:
        image = await asyncio.to_thread(state.t2i, prompt)
        image = await asyncio.to_thread(state.bg, image)
        # Squarize for 3D stability
        sq = await asyncio.to_thread(center_crop_to_square, image)
        outputs = await asyncio.to_thread(state.i23d.infer, sq)
        # Validate
        views = await asyncio.to_thread(state.render_views, outputs)
        ok, (clip_s, aes_s) = await asyncio.to_thread(state.val.validate, prompt, views)
        if not ok:
            logger.warning(f"Validation failed (CLIP={clip_s:.3f}, AES={aes_s:.2f}); retrying once.")
            # retry once
            image = await asyncio.to_thread(state.t2i, prompt)
            image = await asyncio.to_thread(state.bg, image)
            sq = await asyncio.to_thread(center_crop_to_square, image)
            outputs = await asyncio.to_thread(state.i23d.infer, sq)
            views = await asyncio.to_thread(state.render_views, outputs)
            ok, (clip_s, aes_s) = await asyncio.to_thread(state.val.validate, prompt, views)
            if not ok:
                logger.error(f"Second validation failed; returning 204. Scores: CLIP={clip_s:.3f} AES={aes_s:.2f}")
                return Response(status_code=204)

        # Export PLY to buffer
        with tempfile.NamedTemporaryFile(suffix=".ply") as fp:
            await asyncio.to_thread(state.i23d.export_ply, outputs, fp.name)
            data = fp.read()
        logger.info(f"/generate OK in {timer.since('t0'):.1f}s")
        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_video/")
async def generate_video(prompt: str = Form(...)):
    timer = Timer(); timer.stamp("t0")
    try:
        image = await asyncio.to_thread(state.t2i, prompt)
        image = await asyncio.to_thread(state.bg, image)
        sq = await asyncio.to_thread(center_crop_to_square, image)
        outputs = await asyncio.to_thread(state.i23d.infer, sq)

        # Validate like /generate
        views = await asyncio.to_thread(state.render_views, outputs)
        ok, (clip_s, aes_s) = await asyncio.to_thread(state.val.validate, prompt, views)
        if not ok:
            logger.warning(f"Validation failed (CLIP={clip_s:.3f}, AES={aes_s:.2f}); retrying once.")
            image = await asyncio.to_thread(state.t2i, prompt)
            image = await asyncio.to_thread(state.bg, image)
            sq = await asyncio.to_thread(center_crop_to_square, image)
            outputs = await asyncio.to_thread(state.i23d.infer, sq)
            views = await asyncio.to_thread(state.render_views, outputs)
            ok, (clip_s, aes_s) = await asyncio.to_thread(state.val.validate, prompt, views)
            if not ok:
                logger.error(f"Second validation failed; returning 204. Scores: CLIP={clip_s:.3f} AES={aes_s:.2f}")
                return Response(status_code=204)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as fp:
            await asyncio.to_thread(
                state.i23d.render_turntable, outputs, fp.name,
                state.cfg["render"]["turntable_seconds"],
                state.cfg["render"]["fps"],
                state.cfg["render"]["yaw_degrees"],
                state.cfg["render"]["elevation_degrees"]
            )
            data = fp.read()
        logger.info(f"/generate_video OK in {timer.since('t0'):.1f}s")
        return Response(content=data, media_type="video/mp4")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8093)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
