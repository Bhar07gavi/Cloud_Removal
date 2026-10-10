import numpy as np
import torch

def to_model_range(x):
    """Convert uint8 [0, 255] to model range float [-1, 1]."""
    if isinstance(x, torch.Tensor):
        return (x.float() / 127.5) - 1.0
    return (x.astype(np.float32) / 127.5) - 1.0

def to_unit_range(x):
    """Convert model range float [-1, 1] to unit range [0, 1] with clipping."""
    if isinstance(x, torch.Tensor):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def unit_to_model_range(x):
    """Convert unit range [0, 1] to model range float [-1, 1]."""
    return (x * 2.0) - 1.0
