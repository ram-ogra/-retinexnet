"""
EnhanceNet - Illumination Enhancement Network for RetinexNet
Takes the decomposed illumination map and enhances it to produce
a well-lit illumination layer, which is then recombined with
the Reflectance to reconstruct the final enhanced image.

Architecture: U-Net style encoder-decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            padding=kernel_size // 2, bias=not use_bn
        )
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EnhanceNet(nn.Module):
    """
    U-Net style illumination enhancement network.

    Input : concatenation of [Illumination (1ch) + Reflectance (3ch)] = 4 channels
    Output: enhanced illumination map (1 channel), bounded in [0, 1]

    The reflectance is used as a spatial context hint so the network
    preserves structure-aware brightness adjustments.
    """

    def __init__(self, channel: int = 64):
        super(EnhanceNet, self).__init__()

        # ── Encoder (downsampling path) ────────────────────────────────────
        # Input: illumination(1) + reflectance(3) = 4 channels
        self.enc1 = ConvBlock(4,       channel,     use_bn=False)   # full res
        self.enc2 = ConvBlock(channel, channel * 2)                 # 1/2
        self.enc3 = ConvBlock(channel * 2, channel * 4)             # 1/4

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBlock(channel * 4, channel * 8),
            ConvBlock(channel * 8, channel * 8),
        )

        # ── Decoder (upsampling path with skip connections) ───────────────
        self.up3   = nn.ConvTranspose2d(channel * 8, channel * 4, 2, stride=2)
        self.dec3  = ConvBlock(channel * 8, channel * 4)             # 1/4

        self.up2   = nn.ConvTranspose2d(channel * 4, channel * 2, 2, stride=2)
        self.dec2  = ConvBlock(channel * 4, channel * 2)             # 1/2

        self.up1   = nn.ConvTranspose2d(channel * 2, channel, 2, stride=2)
        self.dec1  = ConvBlock(channel * 2, channel)                 # full res

        # ── Output head: 1 channel illumination, bounded [0,1] ───────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, illumination: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            illumination : [B, 1, H, W]   – raw illumination from DecomNet
            reflectance  : [B, 3, H, W]   – raw reflectance from DecomNet
        Returns:
            enhanced_illu: [B, 1, H, W]   – brightness-enhanced illumination
        """
        # Concatenate along channel dim for context-aware enhancement
        x = torch.cat([illumination, reflectance], dim=1)           # [B, 4, H, W]

        # ── Encode ──────────────────────────────────────────────────────
        e1 = self.enc1(x)                                           # [B, C,   H,   W]
        e2 = self.enc2(self.pool(e1))                               # [B, 2C,  H/2, W/2]
        e3 = self.enc3(self.pool(e2))                               # [B, 4C,  H/4, W/4]

        # ── Bottleneck ──────────────────────────────────────────────────
        b  = self.bottleneck(self.pool(e3))                         # [B, 8C,  H/8, W/8]

        # ── Decode with skip connections ────────────────────────────────
        d3 = self.up3(b)                                            # [B, 4C,  H/4, W/4]
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                                           # [B, 2C,  H/2, W/2]
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                                           # [B, C,   H,   W]
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)                                    # [B, 1,   H,   W]

    @staticmethod
    def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle size mismatch between upsampled tensor and skip connection."""
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[2:],
                mode="bilinear", align_corners=False
            )
        return torch.cat([upsampled, skip], dim=1)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EnhanceNet()
    model.eval()
    illu = torch.rand(1, 1, 256, 256)
    refl = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        enhanced = model(illu, refl)
    print(f"EnhanceNet  output: {enhanced.shape}")
    print(f"  enhanced_illu in [{enhanced.min():.3f}, {enhanced.max():.3f}]")
