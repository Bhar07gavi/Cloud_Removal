"""Tiled inference engine for arbitrary resolution satellite imagery."""

from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import to_model_range, to_unit_range


def make_1d_blend_window(size: int, overlap: int, device: torch.device) -> torch.Tensor:
    """Create a 1D trapezoidal/cosine blend ramp.

    Values are 1.0 in the center and smoothly ramp down to 0 at the overlap boundaries.
    """
    if overlap <= 0:
        return torch.ones(size, device=device, dtype=torch.float32)

    ramp = torch.ones(size, device=device, dtype=torch.float32)
    # Ramps use half-cosine for smooth C1-continuous transitions
    x = torch.arange(overlap, device=device, dtype=torch.float32)
    ramp[:overlap] = 0.5 * (1.0 - torch.cos(math.pi * (x + 0.5) / overlap))
    ramp[-overlap:] = torch.flip(ramp[:overlap], dims=[0])
    return ramp


def make_2d_blend_window(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    """Create a 2D separable blending window for (1, 1, tile_size, tile_size)."""
    w1d = make_1d_blend_window(tile_size, overlap, device)
    w2d = torch.outer(w1d, w1d).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Ensure minimum non-zero weight to prevent divide-by-zero
    return torch.clamp(w2d, min=1e-4)


class TiledInferenceEngine:
    """Performs tiled sliding-window inference with overlap blending and edge padding.

    Preserves exact original image dimensions.
    """

    def __init__(
        self,
        generator: nn.Module,
        device: torch.device | str = "cpu",
        tile_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.generator = generator.to(self.device)
        self.generator.eval()
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.batch_size = max(1, int(batch_size))
        self.blend_window = make_2d_blend_window(self.tile_size, self.overlap, self.device)

    def _compute_steps(self, total_dim: int) -> List[int]:
        """Compute top-left 1D coordinates covering total_dim with tile_size and overlap."""
        if total_dim <= self.tile_size:
            return [0]
        step = max(1, self.tile_size - self.overlap)
        coords = list(range(0, total_dim - self.tile_size + 1, step))
        if coords[-1] + self.tile_size < total_dim:
            coords.append(total_dim - self.tile_size)
        return sorted(list(set(coords)))

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Run tiled inference on a normalized PyTorch tensor.

        Args:
            x: Tensor of shape (C, H, W) or (1, C, H, W) with values in [-1, 1].

        Returns:
            Tensor of shape (C, H, W) with values in [-1, 1] preserving (H, W).
        """
        squeeze_batch = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True

        x = x.to(self.device)
        _, channels, orig_h, orig_w = x.shape

        # 1. Determine padding if image dimension is smaller than tile_size
        pad_bottom = max(0, self.tile_size - orig_h)
        pad_right = max(0, self.tile_size - orig_w)
        if pad_bottom > 0 or pad_right > 0:
            # Pad with reflection if feasible, otherwise replicate
            pad_mode = "reflect" if (orig_h > 1 and orig_w > 1 and pad_bottom < orig_h and pad_right < orig_w) else "replicate"
            x_padded = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)
        else:
            x_padded = x

        _, _, padded_h, padded_w = x_padded.shape

        # 2. Compute 2D grid coordinates
        y_coords = self._compute_steps(padded_h)
        x_coords = self._compute_steps(padded_w)

        # 3. Canvases for output accumulator and weight accumulator
        canvas = torch.zeros((1, channels, padded_h, padded_w), device=self.device, dtype=torch.float32)
        weight_canvas = torch.zeros((1, 1, padded_h, padded_w), device=self.device, dtype=torch.float32)

        # 4. Extract and batch tiles
        tile_list: List[Tuple[int, int, torch.Tensor]] = []
        for y in y_coords:
            for xc in x_coords:
                patch = x_padded[:, :, y : y + self.tile_size, xc : xc + self.tile_size]
                tile_list.append((y, xc, patch))

        # 5. Process batches through the generator
        window = self.blend_window  # (1, 1, tile_size, tile_size)
        for i in range(0, len(tile_list), self.batch_size):
            batch_slice = tile_list[i : i + self.batch_size]
            batch_tensors = torch.cat([item[2] for item in batch_slice], dim=0)

            outputs = self.generator(batch_tensors)  # (B, C, tile_size, tile_size) in [-1, 1]

            for idx, (y, xc, _) in enumerate(batch_slice):
                out_tile = outputs[idx : idx + 1]
                weighted_tile = out_tile * window
                canvas[:, :, y : y + self.tile_size, xc : xc + self.tile_size] += weighted_tile
                weight_canvas[:, :, y : y + self.tile_size, xc : xc + self.tile_size] += window

        # 6. Normalize by accumulated weights and crop padding
        output = canvas / torch.clamp(weight_canvas, min=1e-5)
        output = output[:, :, :orig_h, :orig_w]
        output = torch.clamp(output, -1.0, 1.0)

        if squeeze_batch:
            output = output.squeeze(0)
        return output

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> Image.Image:
        """Run tiled inference on a PIL Image.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Reconstructed PIL Image (RGB) with identical dimensions.
        """
        orig_w, orig_h = image.size
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy uint8 -> float32 [-1, 1]
        np_img = np.array(image, dtype=np.uint8)  # (H, W, 3)
        tensor_in = to_model_range(np_img)        # (H, W, 3) in [-1, 1]
        tensor_in = torch.from_numpy(tensor_in).permute(2, 0, 1)  # (3, H, W)

        # Forward pass through tiled engine
        tensor_out = self.predict_tensor(tensor_in).cpu()  # (3, H, W)

        # Denormalize [-1, 1] -> [0, 1]
        unit_tensor = to_unit_range(tensor_out)            # (3, H, W)
        np_out = (unit_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        # Guarantee dimensions match original
        result_img = Image.fromarray(np_out, mode="RGB")
        assert result_img.size == (orig_w, orig_h), f"Dimension mismatch: {result_img.size} vs {(orig_w, orig_h)}"
        return result_img
