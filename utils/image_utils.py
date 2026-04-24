"""
image_utils.py – Image I/O and pre/post-processing helpers for RetinexNet.

All internal tensors use:
    dtype : float32
    range : [0.0, 1.0]
    shape : [B, C, H, W]   (batch first, channel first)
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a float32 RGB array in [0, 1].

    Args:
        path: Path to the image file.
    Returns:
        np.ndarray of shape [H, W, 3], dtype=float32, values in [0, 1]
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def save_image(array: np.ndarray, path: str) -> None:
    """
    Save a float32 [H, W, 3] RGB array to disk.

    Args:
        array: float32 ndarray in [0, 1]
        path : Destination file path (extension determines format).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    uint8   = (clipped * 255).astype(np.uint8)
    bgr     = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversions
# ─────────────────────────────────────────────────────────────────────────────

def numpy_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert [H, W, C] float32 ndarray → [1, C, H, W] float32 tensor on device.
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert [1, C, H, W] or [C, H, W] tensor → [H, W, C] float32 ndarray.
    """
    t = tensor.squeeze(0).detach().cpu()
    if t.ndim == 2:          # single-channel [H, W]
        return t.numpy()
    return t.permute(1, 2, 0).numpy()


def tensor_to_numpy_single(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single-channel [1, 1, H, W] illumination tensor → [H, W] ndarray.
    """
    return tensor.squeeze().detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Padding helpers (models prefer dimensions divisible by 8)
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple]:
    """
    Pad a [B, C, H, W] tensor so H and W are divisible by `multiple`.

    Returns:
        padded tensor, (pad_top, pad_bottom, pad_left, pad_right)
    """
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    padded = torch.nn.functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="reflect",
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor: torch.Tensor, pads: Tuple) -> torch.Tensor:
    """Remove padding added by pad_to_multiple."""
    pad_top, pad_bottom, pad_left, pad_right = pads
    _, _, H, W = tensor.shape

    h_end = H - pad_bottom if pad_bottom > 0 else H
    w_end = W - pad_right  if pad_right  > 0 else W

    return tensor[:, :, pad_top:h_end, pad_left:w_end]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction helper
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(
    reflectance: torch.Tensor,
    enhanced_illumination: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct final enhanced image via element-wise multiplication.
    
    Retinex model: I(x) = R(x) * L(x)
    So Enhanced(x) = R(x) * L_enhanced(x)

    Args:
        reflectance           : [B, 3, H, W]
        enhanced_illumination : [B, 1, H, W]
    Returns:
        enhanced image        : [B, 3, H, W], clamped to [0, 1]
    """
    enhanced = reflectance * enhanced_illumination
    return torch.clamp(enhanced, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def illumination_to_rgb(illu_np: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale [H, W] illumination map to an RGB heatmap [H, W, 3]
    using the INFERNO colourmap for better visual contrast.
    """
    normed = np.clip(illu_np, 0, 1)
    uint8  = (normed * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def side_by_side(img_a: np.ndarray, img_b: np.ndarray, gap: int = 10) -> np.ndarray:
    """
    Concatenate two RGB images side by side with a white gap.
    Both images are assumed to have matching height.
    """
    H = max(img_a.shape[0], img_b.shape[0])

    def pad_height(img):
        if img.shape[0] < H:
            diff = H - img.shape[0]
            img = np.pad(img, ((0, diff), (0, 0), (0, 0)))
        return img

    a = pad_height(img_a)
    b = pad_height(img_b)
    spacer = np.ones((H, gap, 3), dtype=np.float32)
    return np.concatenate([a, spacer, b], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        print("GPU detected →", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("No GPU detected → using CPU")
    return torch.device("cpu")
