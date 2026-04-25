"""
image_utils.py - cv2-FREE version using only PIL + numpy
Compatible with all Python versions on Streamlit Cloud
"""

import os
import numpy as np
import torch
from PIL import Image, ImageFilter
from typing import Tuple


def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def save_image(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(uint8, mode="RGB").save(path)


def numpy_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.squeeze(0).detach().cpu()
    if t.ndim == 2:
        return t.numpy()
    return t.permute(1, 2, 0).numpy()


def tensor_to_numpy_single(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().detach().cpu().numpy()


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple]:
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    padded = torch.nn.functional.pad(
        tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor: torch.Tensor, pads: Tuple) -> torch.Tensor:
    pad_top, pad_bottom, pad_left, pad_right = pads
    _, _, H, W = tensor.shape
    h_end = H - pad_bottom if pad_bottom > 0 else H
    w_end = W - pad_right  if pad_right  > 0 else W
    return tensor[:, :, pad_top:h_end, pad_left:w_end]


def reconstruct(reflectance: torch.Tensor, enhanced_illumination: torch.Tensor) -> torch.Tensor:
    return torch.clamp(reflectance * enhanced_illumination, 0.0, 1.0)


def illumination_to_rgb(illu_np: np.ndarray) -> np.ndarray:
    """Convert grayscale illumination map to INFERNO-style heatmap using numpy only."""
    normed = np.clip(illu_np, 0, 1)
    # Inferno-like colormap: dark purple → red → orange → yellow
    r = np.clip(1.5 * normed - 0.1, 0, 1)
    g = np.clip(2.0 * normed - 1.0, 0, 1)
    b = np.clip(0.5 - 2.0 * np.abs(normed - 0.25), 0, 1)
    return np.stack([r, g, b], axis=2).astype(np.float32)


def resize_image(image_np: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image using PIL — no cv2 needed."""
    h, w = image_np.shape[:2]
    if max(h, w) <= max_dim:
        return image_np
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    pil   = Image.fromarray(uint8).resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def compute_sharpness(arr: np.ndarray) -> float:
    """Laplacian variance for sharpness — PIL-based, no cv2."""
    uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    gray  = Image.fromarray(uint8).convert("L")
    lap   = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float(lap.var())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")