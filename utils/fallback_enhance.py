"""
fallback_enhance.py
Traditional image enhancement used when pretrained weights are not available.
Uses: Gamma correction + CLAHE-style histogram equalization via numpy/PIL
No cv2, no training needed - gives good results on real images.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def clahe_numpy(channel: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """
    Approximate CLAHE (Contrast Limited Adaptive Histogram Equalization)
    using pure numpy - no cv2 needed.
    Works tile by tile for local contrast enhancement.
    """
    h, w   = channel.shape
    ch     = channel.copy()
    th, tw = h // grid, w // grid
    if th < 1 or tw < 1:
        return channel

    for i in range(grid):
        for j in range(grid):
            y1, y2 = i*th, (i+1)*th if i < grid-1 else h
            x1, x2 = j*tw, (j+1)*tw if j < grid-1 else w
            tile    = ch[y1:y2, x1:x2]
            hist, _ = np.histogram(tile, bins=256, range=(0,1))
            # Clip histogram
            excess  = np.sum(np.maximum(hist - clip_limit*tile.size/256, 0))
            hist    = np.minimum(hist, clip_limit*tile.size/256)
            hist   += excess / 256
            # Build CDF
            cdf     = np.cumsum(hist)
            cdf     = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
            # Map tile
            idx     = (tile * 255).astype(np.int32).clip(0, 255)
            ch[y1:y2, x1:x2] = cdf[idx].astype(np.float32)
    return ch


def enhance_traditional(image_np: np.ndarray, gamma: float = 0.7) -> dict:
    """
    Full traditional enhancement pipeline.
    Returns same dict format as neural pipeline.

    Steps:
      1. Compute illumination = max(R,G,B)
      2. Gamma correction on illumination
      3. CLAHE on each channel
      4. Reconstruct enhanced image
      5. Boost saturation slightly
    """
    img = np.clip(image_np, 0, 1).astype(np.float32)

    # ── 1. Decompose: Illumination = max channel ──────────────────────────
    illumination = np.max(img, axis=2)                          # [H,W]
    safe_illu    = np.maximum(illumination[:,:,np.newaxis], 1e-6)
    reflectance  = np.clip(img / safe_illu, 0, 1)              # [H,W,3]

    # ── 2. Gamma correction (brighten illumination) ───────────────────────
    enhanced_illu = np.power(np.clip(illumination, 0, 1), gamma)  # [H,W]

    # ── 3. CLAHE on enhanced illumination ─────────────────────────────────
    enhanced_illu = clahe_numpy(enhanced_illu, clip_limit=3.0, grid=8)
    enhanced_illu = np.clip(enhanced_illu, 0, 1)

    # ── 4. Reconstruct ────────────────────────────────────────────────────
    enhanced = reflectance * enhanced_illu[:,:,np.newaxis]
    enhanced = np.clip(enhanced, 0, 1)

    # ── 5. Saturation boost via PIL ───────────────────────────────────────
    pil_enh = Image.fromarray((enhanced*255).astype(np.uint8))
    pil_enh = ImageEnhance.Color(pil_enh).enhance(1.3)          # +30% saturation
    pil_enh = ImageEnhance.Sharpness(pil_enh).enhance(1.1)      # slight sharpen
    enhanced = np.array(pil_enh, dtype=np.float32) / 255.0

    return {
        "reflectance":           reflectance,
        "illumination":          illumination,
        "enhanced_illumination": enhanced_illu,
        "enhanced":              enhanced,
    }