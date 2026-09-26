"""Smoke test: 1-batch, 1-step CPU forward & backward pass."""

import torch
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.training.losses import GANLoss, PixelLoss

def test_pipeline():
    print("Running Smoke Test on CPU...")
    device = torch.device("cpu")
    batch_size = 2
    channels = 3
    h, w = 256, 256

    # 1. Dummy batch in [-1, 1]
    cloud_batch = torch.randn(batch_size, channels, h, w, device=device).clamp(-1, 1)
    gt_batch = torch.randn(batch_size, channels, h, w, device=device).clamp(-1, 1)

    # 2. Generator
    gen = UNetGenerator(in_channels=channels, out_channels=channels).to(device)
    fake = gen(cloud_batch)
    assert fake.shape == (batch_size, channels, h, w), f"Unexpected fake shape: {fake.shape}"
    print(f"  [PASS] Generator forward: {fake.shape}")

    # 3. Discriminator
    disc = PatchGANDiscriminator(in_channels=channels * 2).to(device)
    pred_real = disc(torch.cat([cloud_batch, gt_batch], dim=1))
    pred_fake = disc(torch.cat([cloud_batch, fake], dim=1))
    assert pred_real.shape == (batch_size, 1, 30, 30), f"Unexpected disc shape: {pred_real.shape}"
    print(f"  [PASS] Discriminator forward: {pred_real.shape}")

    # 4. Losses & Backward
    criterion_adv = GANLoss("bce")
    criterion_l1 = PixelLoss()

    loss_d = (criterion_adv(pred_real, True) + criterion_adv(pred_fake, False)) * 0.5
    loss_g = criterion_adv(pred_fake, True) + 100 * criterion_l1(fake, gt_batch)

    loss_d.backward(retain_graph=True)
    loss_g.backward()
    print("  [PASS] Backward pass executed successfully.")
    print("ALL SMOKE TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
