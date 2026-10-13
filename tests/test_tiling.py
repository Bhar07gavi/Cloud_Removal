"""Tests for tiled sliding-window inference with edge padding and blending."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.generator import UNetGenerator
from src.inference.tiling import (
    TiledInferenceEngine,
    make_1d_blend_window,
    make_2d_blend_window,
)


class IdentityGenerator(nn.Module):
    """Deterministic identity module for testing seamless stitching."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InvertGenerator(nn.Module):
    """Deterministic inverter module for testing color transform stitching."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


def test_blend_window_properties():
    """Verify 1D and 2D blend windows are symmetric, non-zero, and ramp smoothly."""
    device = torch.device("cpu")
    size = 256
    overlap = 32

    w1d = make_1d_blend_window(size, overlap, device)
    assert w1d.shape == (size,)
    assert torch.all(w1d >= 0.0)
    assert torch.all(w1d <= 1.0)
    # Check center is 1.0
    assert torch.allclose(w1d[overlap : size - overlap], torch.tensor(1.0))
    # Check symmetry
    assert torch.allclose(w1d, torch.flip(w1d, dims=[0]))

    w2d = make_2d_blend_window(size, overlap, device)
    assert w2d.shape == (1, 1, size, size)
    assert torch.all(w2d > 0.0)
    assert w2d.min() >= 1e-4


@pytest.mark.parametrize("h,w", [(256, 256), (300, 400), (512, 512), (180, 220), (75, 95)])
def test_identity_stitching_reconstructs_exact_dimensions(h, w):
    """Verify tiled engine preserves exact dimensions and numerical values with IdentityGenerator."""
    engine = TiledInferenceEngine(
        generator=IdentityGenerator(),
        device="cpu",
        tile_size=256,
        overlap=32,
        batch_size=4,
    )

    # Create dummy tensor in [-1, 1]
    x = torch.linspace(-1.0, 1.0, h * w).reshape(1, 1, h, w).repeat(1, 3, 1, 1)
    out = engine.predict_tensor(x)

    assert out.shape == (1, 3, h, w), f"Expected shape {(1, 3, h, w)}, got {out.shape}"
    # Identity must reconstruct very close to input
    assert torch.allclose(out, x, atol=1e-3), "Identity reconstruction failed"


@pytest.mark.parametrize("h,w", [(256, 256), (410, 310), (100, 100), (512, 512)])
def test_pil_image_dimensions_and_channels(h, w):
    """Verify PIL input and output dimensions match identically across various sizes."""
    engine = TiledInferenceEngine(
        generator=IdentityGenerator(),
        device="cpu",
        tile_size=256,
        overlap=32,
    )

    arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    img_in = Image.fromarray(arr, mode="RGB")

    img_out = engine.predict_pil(img_in)
    assert img_out.size == (w, h), f"Expected size {(w, h)}, got {img_out.size}"
    assert img_out.mode == "RGB"


def test_unet_generator_tiled_inference():
    """Verify full UNetGenerator forward pass through tiled inference engine."""
    torch.manual_seed(42)
    unet = UNetGenerator(in_channels=3, out_channels=3)
    engine = TiledInferenceEngine(
        generator=unet,
        device="cpu",
        tile_size=256,
        overlap=32,
        batch_size=2,
    )

    # Test arbitrary size image (350x450)
    test_tensor = torch.randn(3, 350, 450).clamp(-1.0, 1.0)
    out = engine.predict_tensor(test_tensor)

    assert out.shape == (3, 350, 450)
    assert out.min() >= -1.0
    assert out.max() <= 1.0
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("overlap", [0, 16, 32, 64])
def test_configurable_overlap(overlap):
    """Verify that different overlap values execute without error and preserve shape."""
    engine = TiledInferenceEngine(
        generator=IdentityGenerator(),
        device="cpu",
        tile_size=256,
        overlap=overlap,
    )
    arr = np.random.randint(0, 256, size=(300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out = engine.predict_pil(img)
    assert out.size == (300, 300)
