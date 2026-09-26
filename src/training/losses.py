"""Loss functions for pix2pix GAN training."""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Adversarial loss supporting BCE and LSGAN (MSE)."""

    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "bce":
            self.criterion = nn.BCELoss()
        elif self.loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'bce' or 'lsgan'.")

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target_val = 1.0 if is_real else 0.0
        target = torch.full_like(prediction, target_val)
        return self.criterion(prediction, target)


class PixelLoss(nn.Module):
    """L1 pixel reconstruction loss."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(generated, target)
