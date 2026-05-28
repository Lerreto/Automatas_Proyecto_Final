import io
import base64
import torch
import numpy as np
from PIL import Image

from backend.model import NCA

CHANNELS = 32
IMG_SIZE = 100
OUTPUT_SIZE = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 5e-4  # umbral de convergencia RGB

_model = None


def load_model(path: str) -> NCA:
    global _model
    if _model is not None:
        return _model

    checkpoint = torch.load(path, map_location=DEVICE)

    model = NCA(channels=CHANNELS, update_rate=0.5).to(DEVICE)
    model.eval()

    # latest.pth is a full training dict; best.pth may be just the state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and any(k.startswith("perception") or k.startswith("update") for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    _model = model
    return _model


def _pil_to_tensor(img: Image.Image, size: int, device: str) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _mask_pil_to_tensor(mask: Image.Image, size: int, device: str) -> torch.Tensor:
    mask = mask.convert("L").resize((size, size), Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _ssim_zone(out: torch.Tensor, orig: torch.Tensor, zone: torch.Tensor) -> float:
    """SSIM global sobre una zona binaria. out, orig: (1,3,H,W); zone: (1,1,H,W) 1=incluir."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    valid = zone.expand_as(out)          # (1, 3, H, W)
    n = valid.sum() + 1e-8

    mu_x = (out * valid).sum() / n
    mu_y = (orig * valid).sum() / n

    dx = (out - mu_x) * valid
    dy = (orig - mu_y) * valid

    sigma_x2 = dx.pow(2).sum() / n
    sigma_y2 = dy.pow(2).sum() / n
    sigma_xy = (dx * dy).sum() / n

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float((num / den.clamp(min=1e-8)).clamp(0, 1))


def _tensor_to_png_b64(t: torch.Tensor) -> str:
    """Convert (1, C, H, W) or (1, 3, H, W) tensor to upscaled PNG base64."""
    rgb = t[0, :3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8, "RGB")
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def reconstruct(
    img_pil: Image.Image,
    original_pil: Image.Image,
    mask_pil: Image.Image,
    steps: int = 120,
    sample_every: int = 2,
):
    """
    Run the NCA reconstruction loop.

    img_pil:      corrupted image (hole pixels = black)
    original_pil: ground truth (original before painting)
    mask_pil:     binary mask (white=visible, black=hole)

    Returns: (frames, metrics, peak)
      frames:  list of base64 PNG strings, one per sampled frame
      metrics: dict with per-step lists: convergence, activity, psnr_hole, ssim_hole
      peak:    dict with the best frame found by SSIM:
                 frame (b64 PNG), step (1-indexed), psnr, ssim
    """
    model = _model
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img_t      = _pil_to_tensor(img_pil,      IMG_SIZE, DEVICE)  # (1,3,100,100) corrupted
    original_t = _pil_to_tensor(original_pil, IMG_SIZE, DEVICE)  # (1,3,100,100) ground truth
    mask_t     = _mask_pil_to_tensor(mask_pil, IMG_SIZE, DEVICE) # (1,1,100,100)

    # mask_t=1 → VISIBLE, 0 → RECONSTRUCT
    corrupted = img_t * mask_t
    hole      = 1.0 - mask_t  # zona reconstruida — aquí medimos calidad

    metrics = {"convergence": [], "activity": [], "psnr_hole": [], "ssim_hole": []}

    peak = {"frame": None, "step": 0, "psnr": 0.0, "ssim": -1.0}

    with torch.no_grad():
        state = NCA.init_state(corrupted, mask_t, channels=CHANNELS)
        frames = []
        last_valid = state.clone()

        for step in range(steps):
            prev = state.clone()
            state = model(state)

            # re-impose Dirichlet boundary: visible pixels stay as original
            state[:, :3] = state[:, :3] * (1.0 - mask_t) + corrupted * mask_t

            # NaN guard
            if torch.isnan(state).any():
                state = last_valid.clone()
            else:
                last_valid = state.clone()

            # ── Métricas ──────────────────────────────────────────────────────
            # 1. Convergencia solo RGB (canales ocultos vibran por Bernoulli — no sirven)
            delta = (state[:, :3] - prev[:, :3]).abs().mean().item()
            metrics["convergence"].append(round(delta, 6))

            # 2. Células activas RGB con cambio > ε
            active_frac = (
                (state[:, :3] - prev[:, :3]).abs().mean(dim=1) > EPSILON
            ).float().mean().item() * 100
            metrics["activity"].append(round(active_frac, 2))

            # 3. PSNR en zona hueco vs ground truth
            out      = state[:, :3].clamp(0, 1)
            mse_hole = ((out - original_t).pow(2) * hole).sum() / (hole.sum() * 3 + 1e-8)
            psnr     = -10 * torch.log10(mse_hole + 1e-8).item()
            metrics["psnr_hole"].append(round(psnr, 2))

            # 4. SSIM en zona hueco vs ground truth
            ssim = _ssim_zone(out, original_t, hole)
            metrics["ssim_hole"].append(round(ssim, 4))

            # Track peak SSIM frame for direct comparison view
            if ssim > peak["ssim"]:
                peak["ssim"]  = round(ssim, 4)
                peak["psnr"]  = round(psnr, 2)
                peak["step"]  = step + 1
                peak["frame"] = _tensor_to_png_b64(state)

            if (step + 1) % sample_every == 0:
                frames.append(_tensor_to_png_b64(state))

    return frames, metrics, peak
