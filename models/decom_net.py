"""
DecomNet - Decomposition Network for RetinexNet
Decomposes a low-light image into:
  - Reflectance (R): Intrinsic material properties (3-channel)
  - Illumination (I): Lighting conditions (1-channel)

Based on: "Deep Retinex Decomposition for Low-Light Enhancement"
Wei Chen et al., BMVC 2018
"""

import torch
import torch.nn as nn


class DecomNet(nn.Module):
    """
    Shallow convolutional network that decomposes an input image
    into reflectance and illumination components using Retinex theory.

    Architecture:
        - 5 convolutional layers with ReLU activations
        - Final Sigmoid to bound outputs in [0, 1]
        - Input: 4 channels (RGB + max-channel illumination hint)
        - Output: 4 channels split into R (3ch) + I (1ch)
    """

    def __init__(self, num_layers: int = 5, channel: int = 64, kernel_size: int = 3):
        super(DecomNet, self).__init__()

        self.channel = channel

        # ── Build layer stack ──────────────────────────────────────────────
        layers = []

        # First layer: RGB(3) + max-projection(1) = 4 input channels
        layers += [
            nn.Conv2d(4, channel, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
        ]

        # Hidden layers
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ]

        # Output layer → R(3) + I(1), bounded [0,1]
        layers += [
            nn.Conv2d(channel, 4, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input image tensor [B, 3, H, W], values in [0, 1]
        Returns:
            reflectance  [B, 3, H, W] – material colour map
            illumination [B, 1, H, W] – brightness / lighting map
        """
        # Append per-pixel max channel as coarse illumination prior
        max_channel = torch.max(x, dim=1, keepdim=True).values    # [B,1,H,W]
        inp = torch.cat([x, max_channel], dim=1)                   # [B,4,H,W]

        out = self.net(inp)                                         # [B,4,H,W]

        reflectance  = out[:, :3, :, :]                            # [B,3,H,W]
        illumination = out[:, 3:4, :, :]                           # [B,1,H,W]

        return reflectance, illumination


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DecomNet()
    model.eval()
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        R, I = model(dummy)
    print(f"DecomNet  R:{R.shape}  I:{I.shape}")
    print(f"  R in [{R.min():.3f}, {R.max():.3f}]  I in [{I.min():.3f}, {I.max():.3f}]")
