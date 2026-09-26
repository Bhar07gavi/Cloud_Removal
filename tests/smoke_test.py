"""Smoke test: 1-batch, 1-step CPU verification for Generator, Discriminator, and losses.

Runs a quick forward + backward pass with synthetic data to ensure all components
wire together correctly before committing to a real GPU training run.

Usage:
    python -m pytest tests/smoke_test.py -v
    python tests/smoke_test.py
"""

import sys
from pathlib import Path

import torch

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.generator import build_generator
from src.models.discriminator import build_discriminator
from src.training.losses import build_gan_loss


def test_generator_output_shape() -> None:
    """Generator should map (2, 3, 256, 256) → (2, 3, 256, 256)."""
    gen = build_generator(in_channels=3, out_channels=3)
    gen.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = gen(x)
    assert out.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {out.shape}"
    # Output should be bounded by Tanh → [-1, 1]
    assert out.min() >= -1.0 and out.max() <= 1.0, "Generator output out of [-1, 1] range"
    print("[OK] Generator: (2,3,256,256) -> (2,3,256,256), range [-1,1]")


def test_discriminator_output_shape() -> None:
    """Discriminator should map (2, 6, 256, 256) → (2, 1, 30, 30)."""
    disc = build_discriminator(in_channels=6)
    disc.eval()
    cloudy = torch.randn(2, 3, 256, 256)
    candidate = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = disc(cloudy, candidate)
    assert out.shape == (2, 1, 30, 30), f"Expected (2,1,30,30), got {out.shape}"
    # Output should be bounded by Sigmoid → [0, 1]
    assert out.min() >= 0.0 and out.max() <= 1.0, "Discriminator output out of [0, 1] range"
    print("[OK] Discriminator: (2,3,256,256)x2 -> (2,1,30,30), range [0,1]")


def test_loss_computation() -> None:
    """Loss functions should compute without errors and produce scalar tensors."""
    gen = build_generator()
    disc = build_discriminator()
    criterion = build_gan_loss(lambda_l1=100.0)

    cloudy = torch.randn(2, 3, 256, 256)
    gt = torch.randn(2, 3, 256, 256).clamp(-1, 1)

    generated = gen(cloudy)

    pred_real = disc(cloudy, gt)
    pred_fake = disc(cloudy, generated.detach())

    d_loss = criterion.discriminator_loss(pred_real, pred_fake)
    assert d_loss.dim() == 0, f"D-loss should be scalar, got dim={d_loss.dim()}"
    assert d_loss.item() > 0, "D-loss should be positive"

    pred_fake_g = disc(cloudy, generated)
    g_total, g_adv, g_l1 = criterion.generator_loss(pred_fake_g, generated, gt)
    assert g_total.dim() == 0, f"G-total should be scalar, got dim={g_total.dim()}"
    assert g_adv.dim() == 0, f"G-adv should be scalar, got dim={g_adv.dim()}"
    assert g_l1.dim() == 0, f"G-L1 should be scalar, got dim={g_l1.dim()}"

    print(f"[OK] Losses -- D: {d_loss.item():.4f}, G_total: {g_total.item():.4f}, "
          f"G_adv: {g_adv.item():.4f}, G_L1: {g_l1.item():.4f}")


def test_backward_pass() -> None:
    """Full forward + backward pass should complete without errors."""
    gen = build_generator()
    disc = build_discriminator()
    criterion = build_gan_loss(lambda_l1=100.0)

    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    cloudy = torch.randn(2, 3, 256, 256)
    gt = torch.randn(2, 3, 256, 256).clamp(-1, 1)

    # Train D
    generated = gen(cloudy)
    pred_real = disc(cloudy, gt)
    pred_fake = disc(cloudy, generated.detach())
    d_loss = criterion.discriminator_loss(pred_real, pred_fake)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # Train G
    pred_fake_g = disc(cloudy, generated)
    g_loss, _, _ = criterion.generator_loss(pred_fake_g, generated, gt)
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    print("[OK] Full backward pass (1 D-step + 1 G-step) completed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("  Smoke Test — Cloud Removal GAN")
    print("=" * 60)
    test_generator_output_shape()
    test_discriminator_output_shape()
    test_loss_computation()
    test_backward_pass()
    print("=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60)
