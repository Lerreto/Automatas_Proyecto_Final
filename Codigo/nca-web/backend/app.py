import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import inference

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "latest.pth"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CHECKPOINT_PATH.exists():
        inference.load_model(str(CHECKPOINT_PATH))
        print(f"[NCA] Model loaded from {CHECKPOINT_PATH} on {inference.DEVICE}")
    else:
        print(f"[NCA] WARNING: checkpoint not found at {CHECKPOINT_PATH}")
        print("[NCA] Place latest.pth in backend/checkpoints/ and restart.")
    yield


app = FastAPI(title="NCA Reconstruction", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReconstructRequest(BaseModel):
    image:    str = Field(..., description="Base64 PNG of the corrupted image (hole=black)")
    original: str = Field(..., description="Base64 PNG of the original image (no hole)")
    mask:     str = Field(..., description="Base64 PNG of the binary mask (white=visible)")
    steps:        int = Field(default=120, ge=10, le=400)
    sample_every: int = Field(default=2, ge=1, le=10)


class Metrics(BaseModel):
    convergence: list[float]
    activity:    list[float]
    psnr_hole:   list[float]
    ssim_hole:   list[float]


class Peak(BaseModel):
    frame: str    = Field(..., description="Base64 PNG of the best-SSIM frame")
    step:  int    = Field(..., description="1-indexed step where the peak occurred")
    psnr:  float
    ssim:  float


class ReconstructResponse(BaseModel):
    frames:  list[str]
    count:   int
    device:  str
    steps:   int
    metrics: Metrics
    peak:    Peak


def _decode_b64_image(data: str):
    from PIL import Image
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw))


@app.get("/api/health")
def health():
    return {"status": "ok", "device": inference.DEVICE}


@app.post("/api/reconstruct", response_model=ReconstructResponse)
def reconstruct(req: ReconstructRequest):
    if inference._model is None:
        if CHECKPOINT_PATH.exists():
            inference.load_model(str(CHECKPOINT_PATH))
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found. Place latest.pth in backend/checkpoints/"
            )

    try:
        img_pil      = _decode_b64_image(req.image)
        original_pil = _decode_b64_image(req.original)
        mask_pil     = _decode_b64_image(req.mask)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode images: {e}")

    try:
        frames, metrics, peak = inference.reconstruct(
            img_pil, original_pil, mask_pil, req.steps, req.sample_every
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")

    return ReconstructResponse(
        frames=frames,
        count=len(frames),
        device=inference.DEVICE,
        steps=req.steps,
        metrics=Metrics(**metrics),
        peak=Peak(**peak),
    )


# Serve frontend — mounted AFTER API routes so /api/* takes priority
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
