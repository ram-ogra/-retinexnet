"""
inference.py – CLI inference script for RetinexNet.

Usage:
    # Enhance a single image
    python inference.py --input path/to/dark.jpg --output results/enhanced.jpg

    # Batch process a directory
    python inference.py --input_dir data/low/ --output_dir results/

    # Show intermediate maps
    python inference.py --input dark.jpg --save_maps

Run `python inference.py --help` for full options.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import torch
import cv2

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models.decom_net   import DecomNet
from models.enhance_net import EnhanceNet
from utils.image_utils  import (
    load_image, save_image,
    numpy_to_tensor, tensor_to_numpy, tensor_to_numpy_single,
    pad_to_multiple, unpad,
    reconstruct,
    illumination_to_rgb,
    get_device,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR   = os.path.join(os.path.dirname(__file__), "weights")
DECOM_WEIGHTS = os.path.join(WEIGHTS_DIR, "pretrained_decom.pth")
ENHANCE_WEIGHTS = os.path.join(WEIGHTS_DIR, "pretrained_enhance.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_models(device: torch.device):
    """
    Load DecomNet + EnhanceNet from pretrained weights.
    Falls back to random initialisation if weights are not found,
    and prompts the user to run download_weights.py first.
    """
    decom_net   = DecomNet().to(device)
    enhance_net = EnhanceNet().to(device)

    # ── DecomNet ──────────────────────────────────────────────────────────
    if os.path.isfile(DECOM_WEIGHTS):
        state = torch.load(DECOM_WEIGHTS, map_location=device)
        decom_net.load_state_dict(state, strict=False)
        print(f"✅ Loaded DecomNet   ← {DECOM_WEIGHTS}")
    else:
        print(f"⚠️  DecomNet weights not found at {DECOM_WEIGHTS}")
        print("   Run: python download_weights.py")
        print("   Continuing with random weights (demo mode)…")

    # ── EnhanceNet ────────────────────────────────────────────────────────
    if os.path.isfile(ENHANCE_WEIGHTS):
        state = torch.load(ENHANCE_WEIGHTS, map_location=device)
        enhance_net.load_state_dict(state, strict=False)
        print(f"✅ Loaded EnhanceNet ← {ENHANCE_WEIGHTS}")
    else:
        print(f"⚠️  EnhanceNet weights not found at {ENHANCE_WEIGHTS}")
        print("   Run: python download_weights.py")
        print("   Continuing with random weights (demo mode)…")

    decom_net.eval()
    enhance_net.eval()
    return decom_net, enhance_net


# ─────────────────────────────────────────────────────────────────────────────
# Core inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def enhance_image(
    image_np:    np.ndarray,
    decom_net:   DecomNet,
    enhance_net: EnhanceNet,
    device:      torch.device,
    gamma:       float = 1.0,
):
    """
    Full RetinexNet inference pipeline.

    Steps:
        1. Convert image to tensor and pad to model-friendly size.
        2. DecomNet → Reflectance + Illumination.
        3. EnhanceNet → Enhanced Illumination.
        4. Apply optional gamma correction.
        5. Reconstruct: Enhanced = Reflectance * Enhanced_Illumination.
        6. Unpad and return all intermediate maps.

    Args:
        image_np   : float32 [H, W, 3] input image in [0, 1]
        decom_net  : loaded DecomNet
        enhance_net: loaded EnhanceNet
        device     : torch device
        gamma      : extra gamma boost (1.0 = no change, < 1 = brighter)

    Returns dict with keys:
        original, reflectance, illumination, enhanced_illumination, enhanced
        (all as float32 numpy arrays)
    """
    t0 = time.time()

    # ── 1. Prepare tensor ─────────────────────────────────────────────────
    tensor = numpy_to_tensor(image_np, device)           # [1, 3, H, W]
    padded, pads = pad_to_multiple(tensor, multiple=8)

    # ── 2. Decompose ──────────────────────────────────────────────────────
    refl, illu = decom_net(padded)                        # [1,3,H,W], [1,1,H,W]

    # ── 3. Enhance illumination ───────────────────────────────────────────
    enhanced_illu = enhance_net(illu, refl)               # [1, 1, H, W]

    # ── 4. Optional gamma boost (applied to illumination map) ─────────────
    if gamma != 1.0:
        enhanced_illu = torch.clamp(enhanced_illu ** gamma, 0.0, 1.0)

    # ── 5. Reconstruct ────────────────────────────────────────────────────
    enhanced = reconstruct(refl, enhanced_illu)           # [1, 3, H, W]

    # ── 6. Unpad all maps ─────────────────────────────────────────────────
    refl          = unpad(refl, pads)
    illu          = unpad(illu, pads)
    enhanced_illu = unpad(enhanced_illu, pads)
    enhanced      = unpad(enhanced, pads)

    elapsed = time.time() - t0

    # ── Convert to numpy ──────────────────────────────────────────────────
    results = {
        "original":             image_np,
        "reflectance":          tensor_to_numpy(refl),
        "illumination":         tensor_to_numpy_single(illu),
        "enhanced_illumination": tensor_to_numpy_single(enhanced_illu),
        "enhanced":             tensor_to_numpy(enhanced),
        "elapsed_ms":           elapsed * 1000,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-image entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_single(args, decom_net, enhance_net, device):
    print(f"\n🖼  Processing: {args.input}")
    image_np = load_image(args.input)
    print(f"   Image size : {image_np.shape[1]}×{image_np.shape[0]} px")

    results = enhance_image(image_np, decom_net, enhance_net, device, gamma=args.gamma)

    # ── Save enhanced image ───────────────────────────────────────────────
    save_image(results["enhanced"], args.output)
    print(f"✅ Enhanced image saved → {args.output}  ({results['elapsed_ms']:.1f} ms)")

    # ── Optionally save intermediate maps ─────────────────────────────────
    if args.save_maps:
        base, ext = os.path.splitext(args.output)

        refl_path  = f"{base}_reflectance{ext}"
        illu_path  = f"{base}_illumination{ext}"
        eillu_path = f"{base}_enhanced_illumination{ext}"

        save_image(results["reflectance"], refl_path)
        save_image(
            illumination_to_rgb(results["illumination"]),
            illu_path,
        )
        save_image(
            illumination_to_rgb(results["enhanced_illumination"]),
            eillu_path,
        )
        print(f"   Maps saved: {refl_path}, {illu_path}, {eillu_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing entry point
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def process_batch(args, decom_net, enhance_net, device):
    input_dir  = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"⚠️  No supported images found in {input_dir}")
        return

    print(f"\n📂 Batch mode: {len(files)} images → {output_dir}")
    total_ms = 0.0

    for idx, fname in enumerate(sorted(files), 1):
        inp_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        try:
            image_np = load_image(inp_path)
            results  = enhance_image(image_np, decom_net, enhance_net, device, gamma=args.gamma)
            save_image(results["enhanced"], out_path)
            total_ms += results["elapsed_ms"]
            print(f"  [{idx:3d}/{len(files)}] {fname:40s}  {results['elapsed_ms']:6.1f} ms")
        except Exception as e:
            print(f"  [{idx:3d}/{len(files)}] ❌ FAILED: {fname}  –  {e}")

    print(f"\n✅ Batch complete. Avg inference time: {total_ms / len(files):.1f} ms/image")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinexNet Low-Light Image Enhancer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single image
    parser.add_argument("--input",   type=str, default=None,
                        help="Path to a single input image.")
    parser.add_argument("--output",  type=str, default="results/enhanced.jpg",
                        help="Path to save the enhanced image.")

    # Batch processing
    parser.add_argument("--input_dir",  type=str, default=None,
                        help="Directory of input images (batch mode).")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save batch-enhanced images.")

    # Options
    parser.add_argument("--gamma",     type=float, default=1.0,
                        help="Gamma exponent applied to illumination map "
                             "(<1 = brighter, >1 = darker).")
    parser.add_argument("--save_maps", action="store_true",
                        help="Also save reflectance and illumination maps.")
    parser.add_argument("--no_gpu",    action="store_true",
                        help="Force CPU even if CUDA is available.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cpu") if args.no_gpu else get_device()

    print("=" * 60)
    print(" RetinexNet Low-Light Enhancer – Inference")
    print("=" * 60)
    print(f" Device : {device}")

    # Load models
    decom_net, enhance_net = load_models(device)

    # Route to single or batch mode
    if args.input_dir:
        process_batch(args, decom_net, enhance_net, device)
    elif args.input:
        process_single(args, decom_net, enhance_net, device)
    else:
        print("\n❌  Provide either --input (single image) or --input_dir (batch).")
        print("    Example:  python inference.py --input dark.jpg")
        sys.exit(1)


if __name__ == "__main__":
    main()
