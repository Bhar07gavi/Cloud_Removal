import numpy as np
import torch
import pytest
from src.data.transforms import to_model_range, to_unit_range, unit_to_model_range

def test_round_trip():
    # numpy
    x_np = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    model_np = to_model_range(x_np)
    unit_np = to_unit_range(model_np)
    round_np = (unit_np * 255.0)
    assert np.allclose(x_np, round_np, atol=1.0)
    
    # torch
    x_pt = torch.randint(0, 256, size=(10, 10), dtype=torch.uint8)
    model_pt = to_model_range(x_pt)
    unit_pt = to_unit_range(model_pt)
    round_pt = (unit_pt * 255.0)
    assert torch.allclose(x_pt.float(), round_pt, atol=1.0)

def test_to_model_range_bounds():
    x_np = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    model_np = to_model_range(x_np)
    assert model_np.min() >= -1.0
    assert model_np.max() <= 1.0
    
    x_pt = torch.randint(0, 256, size=(10, 10), dtype=torch.uint8)
    model_pt = to_model_range(x_pt)
    assert model_pt.min() >= -1.0
    assert model_pt.max() <= 1.0

def test_to_unit_range_clips():
    # Numpy
    x_np = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    unit_np = to_unit_range(x_np)
    assert np.allclose(unit_np, [0.0, 0.5, 1.0])
    
    # Torch
    x_pt = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    unit_pt = to_unit_range(x_pt)
    assert torch.allclose(unit_pt, torch.tensor([0.0, 0.5, 1.0]))

def test_works_on_both_types():
    x_np = np.array([127], dtype=np.uint8)
    res_np = to_model_range(x_np)
    assert isinstance(res_np, np.ndarray)
    
    x_pt = torch.tensor([127], dtype=torch.uint8)
    res_pt = to_model_range(x_pt)
    assert isinstance(res_pt, torch.Tensor)

def test_unit_to_model_range():
    # Numpy
    x_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    model_np = unit_to_model_range(x_np)
    assert np.allclose(model_np, [-1.0, 0.0, 1.0])
    
    # Torch
    x_pt = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    model_pt = unit_to_model_range(x_pt)
    assert torch.allclose(model_pt, torch.tensor([-1.0, 0.0, 1.0]))
    
    # Inverts to_unit_range (in range)
    x_orig = np.random.uniform(-1.0, 1.0, size=(10, 10)).astype(np.float32)
    unit_np = to_unit_range(x_orig)
    inv_np = unit_to_model_range(unit_np)
    assert np.allclose(x_orig, inv_np, atol=1e-6)
