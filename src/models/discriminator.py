"""70×70 PatchGAN Discriminator for conditional image generation.

Architecture: 5-layer convolutional network producing per-patch real/fake maps.
Input:  Concatenated [cloudy_input, candidate] → (N, 6, 256, 256)
Output: Per-patch probability map → (N, 1, 30, 30)
"""

import torch
import torch.nn as nn

__all__ = ["PatchGANDiscriminator", "build_discriminator"]


class DiscriminatorBlock(nn.Module):
    """Discriminator conv block: Conv2d → [BatchNorm] → LeakyReLU(0.2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """70×70 receptive-field PatchGAN discriminator.

    Layer progression for 256×256 input:
        layer1 (stride 2): 256 → 128  [C64,  no BN]
        layer2 (stride 2): 128 → 64   [C128]
        layer3 (stride 2): 64  → 32   [C256]
        layer4 (stride 1): 32  → 31   [C512]
        layer5 (stride 1): 31  → 30   [1-channel output + Sigmoid]
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.layer1 = DiscriminatorBlock(in_channels, 64, stride=2, use_batchnorm=False)
        self.layer2 = DiscriminatorBlock(64, 128, stride=2)
        self.layer3 = DiscriminatorBlock(128, 256, stride=2)
        self.layer4 = DiscriminatorBlock(256, 512, stride=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cloudy_input: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Forward pass on a conditional image pair.

        Args:
            cloudy_input: Cloudy satellite image, shape (N, 3, 256, 256).
            candidate: Real or generated image, shape (N, 3, 256, 256).

        Returns:
            Per-patch real/fake probability map, shape (N, 1, 30, 30).
        """
        x = torch.cat([cloudy_input, candidate], dim=1)  # (N, 6, 256, 256)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)


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


def build_discriminator(in_channels: int = 6) -> PatchGANDiscriminator:
    """Construct a weight-initialized PatchGAN discriminator.

    Args:
        in_channels: Number of input channels (default 6 = 3 cloudy + 3 candidate).

    Returns:
        Initialized PatchGANDiscriminator instance.
    """
    disc = PatchGANDiscriminator(in_channels)
    disc.apply(init_weights)
    return disc
