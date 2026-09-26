"""Pix2Pix U-Net Generator with skip connections for image-to-image translation.

Architecture: 8-stage encoder-decoder with skip connections.
Input:  (N, 3, 256, 256)  cloudy satellite image
Output: (N, 3, 256, 256)  reconstructed cloud-free image, values in [-1, 1]
"""

import torch
import torch.nn as nn

__all__ = ["UNetGenerator", "build_generator"]


class UNetDownBlock(nn.Module):
    """Encoder block: Conv2d (stride 2) → [BatchNorm] → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Decoder block: ConvTranspose2d (stride 2) → BatchNorm → [Dropout] → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net Generator with 8-layer encoder-decoder and skip connections.

    Spatial progression (256 input):
        Encoder: 256→128→64→32→16→8→4→2→1  (bottleneck)
        Decoder: 1→2→4→8→16→32→64→128→256  (output)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()

        # ---- Encoder ----
        self.e1 = UNetDownBlock(in_channels, 64, use_batchnorm=False)  # 256 → 128
        self.e2 = UNetDownBlock(64, 128)                                # 128 → 64
        self.e3 = UNetDownBlock(128, 256)                               # 64  → 32
        self.e4 = UNetDownBlock(256, 512)                               # 32  → 16
        self.e5 = UNetDownBlock(512, 512)                               # 16  → 8
        self.e6 = UNetDownBlock(512, 512)                               # 8   → 4
        self.e7 = UNetDownBlock(512, 512)                               # 4   → 2
        self.e8 = UNetDownBlock(512, 512, use_batchnorm=False)          # 2   → 1  (bottleneck)

        # ---- Decoder (input channels doubled by skip connections except d8) ----
        self.d8 = UNetUpBlock(512, 512, use_dropout=True)               # 1   → 2
        self.d7 = UNetUpBlock(1024, 512, use_dropout=True)              # 2   → 4   (cat with e7)
        self.d6 = UNetUpBlock(1024, 512, use_dropout=True)              # 4   → 8   (cat with e6)
        self.d5 = UNetUpBlock(1024, 512)                                # 8   → 16  (cat with e5)
        self.d4 = UNetUpBlock(1024, 256)                                # 16  → 32  (cat with e4)
        self.d3 = UNetUpBlock(512, 128)                                 # 32  → 64  (cat with e3)
        self.d2 = UNetUpBlock(256, 64)                                  # 64  → 128 (cat with e2)

        # ---- Final layer ----
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )  # 128 → 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections.

        Args:
            x: Cloudy input image, shape (N, C, 256, 256).

        Returns:
            Reconstructed image, shape (N, C, 256, 256), values in [-1, 1].
        """
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)  # bottleneck: (N, 512, 1, 1)

        # Decoder with skip connections
        d8 = self.d8(e8)
        d7 = self.d7(torch.cat([d8, e7], dim=1))
        d6 = self.d6(torch.cat([d7, e6], dim=1))
        d5 = self.d5(torch.cat([d6, e5], dim=1))
        d4 = self.d4(torch.cat([d5, e4], dim=1))
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))

        return self.final(torch.cat([d2, e1], dim=1))


def init_weights(model: nn.Module) -> None:
    """Initialize model weights using the pix2pix convention.

    - Conv2d / ConvTranspose2d: N(0, 0.02)
    - BatchNorm2d: weight ~ N(1.0, 0.02), bias = 0
    """
    for m in model.modules():
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


def build_generator(in_channels: int = 3, out_channels: int = 3) -> UNetGenerator:
    """Construct a weight-initialized U-Net generator.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        out_channels: Number of output image channels.

    Returns:
        Initialized UNetGenerator instance.
    """
    gen = UNetGenerator(in_channels, out_channels)
    gen.apply(init_weights)
    return gen
