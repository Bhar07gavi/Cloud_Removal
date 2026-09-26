"""Loss functions for Pix2Pix GAN training: adversarial (BCE) + L1 reconstruction."""

import torch
import torch.nn as nn

__all__ = ["GANLoss", "build_gan_loss"]


class GANLoss(nn.Module):
    """Loss module for Pix2Pix GAN training combining adversarial BCE and L1 reconstruction loss."""

    def __init__(self, lambda_l1: float = 100.0) -> None:
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def discriminator_loss(
        self, pred_real: torch.Tensor, pred_fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator adversarial loss.

        Args:
            pred_real: Discriminator predictions on real image pairs.
            pred_fake: Discriminator predictions on fake image pairs.

        Returns:
            Averaged BCE loss for real and fake predictions.
        """
        real_loss = self.bce_loss(pred_real, torch.ones_like(pred_real))
        fake_loss = self.bce_loss(pred_fake, torch.zeros_like(pred_fake))
        return (real_loss + fake_loss) * 0.5

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator adversarial and L1 reconstruction loss.

        Args:
            pred_fake: Discriminator predictions on fake image pairs.
            generated: Generated images from generator.
            target: Ground truth target images.

        Returns:
            Tuple of (total_loss, adv_loss, l1_loss).
        """
        adv_loss = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        l1_loss = self.l1_loss(generated, target)
        total = adv_loss + self.lambda_l1 * l1_loss
        return total, adv_loss, l1_loss


def build_gan_loss(lambda_l1: float = 100.0) -> GANLoss:
    """Build and return a GANLoss instance with the given L1 penalty weight.

    Args:
        lambda_l1: Weight for the L1 reconstruction loss term.

    Returns:
        Configured GANLoss instance.
    """
    return GANLoss(lambda_l1=lambda_l1)
