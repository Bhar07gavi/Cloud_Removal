import pytest
import torch
from unittest.mock import patch
from src.utils.device import get_device

def test_device_cpu():
    device = get_device("cpu")
    assert device.type == "cpu"

def test_device_auto_cuda_available():
    with patch('torch.cuda.is_available', return_value=True):
        device = get_device("auto")
        assert device.type == "cuda"
        
        device = get_device("cuda")
        assert device.type == "cuda"

def test_device_auto_cuda_unavailable():
    with patch('torch.cuda.is_available', return_value=False):
        device = get_device("auto")
        assert device.type == "cpu"
        
        device = get_device("cuda")
        assert device.type == "cpu"

def test_device_invalid():
    with pytest.raises(ValueError, match="Invalid device 'tpu'. Valid options are 'auto', 'cuda', 'cpu'."):
        get_device("tpu")
