"""U-Net Generator for pix2pix Cloud Removal."""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """pix2pix U-Net Generator with 8 encoder & 8 decoder blocks + skip connections.

    Input:  (N, in_channels, 256, 256)
    Output: (N, out_channels, 256, 256) with values in [-1, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder (Downsampling: 256 -> 1)
        self.enc1 = self._enc_block(in_channels, 64, use_bn=False)  # 256 -> 128
        self.enc2 = self._enc_block(64, 128)                        # 128 -> 64
        self.enc3 = self._enc_block(128, 256)                       # 64 -> 32
        self.enc4 = self._enc_block(256, 512)                       # 32 -> 16
        self.enc5 = self._enc_block(512, 512)                       # 16 -> 8
        self.enc6 = self._enc_block(512, 512)                       # 8 -> 4
        self.enc7 = self._enc_block(512, 512)                       # 4 -> 2
        self.bottleneck = self._enc_block(512, 512, use_bn=False)   # 2 -> 1

        # Decoder (Upsampling: 1 -> 256 with skip connections)
        self.dec7 = self._dec_block(512, 512, use_dropout=True)     # 1 -> 2
        self.dec6 = self._dec_block(1024, 512, use_dropout=True)    # 2 -> 4
        self.dec5 = self._dec_block(1024, 512, use_dropout=True)    # 4 -> 8
        self.dec4 = self._dec_block(1024, 512)                      # 8 -> 16
        self.dec3 = self._dec_block(1024, 256)                      # 16 -> 32
        self.dec2 = self._dec_block(512, 128)                       # 32 -> 64
        self.dec1 = self._dec_block(256, 64)                        # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _enc_block(in_c: int, out_c: int, use_bn: bool = True) -> nn.Sequential:
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _dec_block(in_c: int, out_c: int, use_dropout: bool = False) -> nn.Sequential:
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b = self.bottleneck(e7)

        d7 = torch.cat([self.dec7(b), e7], dim=1)
        d6 = torch.cat([self.dec6(d7), e6], dim=1)
        d5 = torch.cat([self.dec5(d6), e5], dim=1)
        d4 = torch.cat([self.dec4(d5), e4], dim=1)
        d3 = torch.cat([self.dec3(d4), e3], dim=1)
        d2 = torch.cat([self.dec2(d3), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)

        return self.final(d1)
