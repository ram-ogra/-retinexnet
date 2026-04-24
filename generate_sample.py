"""
generate_sample.py – Creates a synthetic low-light test image.

Useful for testing the pipeline without a real dark photo.
Generates: sample_images/test_dark.jpg
"""

import os
import cv2
import numpy as np


def make_dark_image(out_path: str = "sample_images/test_dark.jpg",
                    size: int = 512, seed: int = 42):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Base gradient ─────────────────────────────────────────────────────
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base  = np.stack([
        0.4 + 0.4 * np.sin(2 * np.pi * x * 2),
        0.3 + 0.4 * np.cos(2 * np.pi * y * 1.5),
        0.5 + 0.3 * np.sin(np.pi * (x + y)),
    ], axis=2)

    # ── Simulated scene objects (coloured blobs) ──────────────────────────
    for _ in range(12):
        cx, cy = rng.integers(60, size - 60, size=2)
        r      = rng.integers(30, 80)
        color  = rng.random(3)
        Y, X   = np.ogrid[:size, :size]
        mask   = (X - cx)**2 + (Y - cy)**2 < r**2
        base[mask] = 0.4 * base[mask] + 0.6 * color

    # ── Apply dark lighting (low average brightness ~0.15) ────────────────
    vignette = 1 - 0.6 * ((x - 0.5)**2 + (y - 0.5)**2) * 2
    vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]

    dark  = base * vignette * 0.25
    dark += rng.normal(0, 0.008, dark.shape)          # slight noise
    dark  = np.clip(dark, 0, 1)

    uint8 = (dark * 255).astype(np.uint8)
    bgr   = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    print(f"✅ Sample dark image saved → {out_path}")
    print(f"   Size: {size}×{size}  |  Avg brightness: {dark.mean()*100:.1f}/100")


if __name__ == "__main__":
    make_dark_image()
