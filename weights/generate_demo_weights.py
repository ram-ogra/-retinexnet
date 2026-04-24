"""
generate_demo_weights.py
Standalone script to generate random-init demo weights.
Run this if download_weights.py cannot reach the internet.

Usage:
    cd retinexnet
    python weights/generate_demo_weights.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.decom_net   import DecomNet
from models.enhance_net import EnhanceNet

OUT_DIR = os.path.dirname(__file__)

decom   = DecomNet()
enhance = EnhanceNet()

torch.save(decom.state_dict(),   os.path.join(OUT_DIR, "pretrained_decom.pth"))
torch.save(enhance.state_dict(), os.path.join(OUT_DIR, "pretrained_enhance.pth"))

print("Demo weights saved:")
print(f"  {os.path.join(OUT_DIR, 'pretrained_decom.pth')}")
print(f"  {os.path.join(OUT_DIR, 'pretrained_enhance.pth')}")
print()
print("NOTE: These are RANDOM weights — output will look noisy.")
print("      Replace with real weights for meaningful results.")
