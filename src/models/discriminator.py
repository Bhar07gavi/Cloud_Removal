"""PatchGAN Discriminator for pix2pix Cloud Removal."""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 receptive field PatchGAN Discriminator.

    Takes concatenated input: [cloudy_input, candidate_image]
    Input:  (N, 2 * channels, 256, 256)
    Output: (N, 1, 30, 30) patch probability map
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: no BatchNorm
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 (stride 1 for 70x70 receptive field)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: Output map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
