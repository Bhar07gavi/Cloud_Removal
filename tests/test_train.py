import pytest
import numpy as np
import yaml
import torch
import csv
from pathlib import Path
import os

from src.training.train import train, train_one_run

@pytest.fixture
def setup_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    
    # 1. Create dataset files
    ds_dir = tmp_path / "dataset"
    cloud_dir = ds_dir / "cloud"
    gt_dir = ds_dir / "gt"
    splits_dir = ds_dir / "splits"
    
    for d in [cloud_dir, gt_dir, splits_dir]:
        d.mkdir(parents=True)
        
    records = []
    # 4 train patches, 2 val patches
    for i in range(6):
        patch_name = f"img_{i}_patch"
        c_path = cloud_dir / f"{patch_name}.npy"
        g_path = gt_dir / f"{patch_name}.npy"
        
        c_data = np.random.uniform(-1.0, 1.0, size=(256, 256, 3)).astype(np.float32)
        g_data = np.random.uniform(-1.0, 1.0, size=(256, 256, 3)).astype(np.float32)
        
        np.save(str(c_path), c_data)
        np.save(str(g_path), g_data)
        
        records.append((f"img_{i}", patch_name, str(c_path).replace('\\', '/'), str(g_path).replace('\\', '/')))

    # Write splits
    train_records = records[:4]
    val_records = records[4:]
    
    train_csv = splits_dir / "test_ds_train.csv"
    val_csv = splits_dir / "test_ds_val.csv"
    
    for csv_path, split_recs in [(train_csv, train_records), (val_csv, val_records)]:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "patch_name", "cloud_path", "gt_path"])
            writer.writerows(split_recs)

    # 2. Write configs
    ds_cfg_path = tmp_path / "dataset.yaml"
    ds_cfg = {
        "name": "test_ds",
        "patch_size": 256,
        "bands": {"count": 3},
        "paths": {
            "raw": "",
            "processed": str(ds_dir).replace('\\', '/'),
            "splits_dir": str(splits_dir).replace('\\', '/'),
            "pairs_manifest": str(ds_dir / "manifest.csv").replace('\\', '/')
        },
        "split": {"train": 0.6, "val": 0.4, "test": 0.0, "seed": 42},
        "normalization": {}
    }
    with open(ds_cfg_path, "w") as f:
        yaml.dump(ds_cfg, f)
        
    model_dir = tmp_path / "model_out"
    model_dir.mkdir()
    
    train_cfg_path = tmp_path / "train.yaml"
    train_cfg = {
        "experiment": {
            "name": "test_exp",
            "model_dir": str(model_dir).replace('\\', '/'),
        },
        "dataset": {
            "config_path": str(ds_cfg_path).replace('\\', '/')
        },
        "loss": {
            "adversarial": "bce",
            "lambda_l1": 100
        },
        "optimizer": {
            "type": "adam",
            "lr": 0.0002,
            "betas": [0.5, 0.999]
        },
        "training": {
            "device": "cpu",
            "seed": 42,
            "epochs": 3,
            "batch_size": 2,
            "num_workers": 0,
            "sample_grid_every_n_epochs": 100
        },
        "mlflow": {
            "tracking_uri": f"file:///{str(tmp_path / 'mlruns').replace(':', '|')}",
            "experiment_name": "test_exp"
        }
    }
    with open(train_cfg_path, "w") as f:
        yaml.dump(train_cfg, f)
        
    return str(train_cfg_path), train_cfg

def test_checkpoint_selection_A(setup_configs, monkeypatch):
    """Sequence: 0.5, 0.3, 0.4 -> best epoch is 2"""
    cfg_path, _ = setup_configs
    
    val_sequence = [0.5, 0.3, 0.4]
    call_idx = 0
    def mock_validate(gen, loader, device, criterion_l1):
        nonlocal call_idx
        v = val_sequence[call_idx]
        call_idx += 1
        return v
        
    monkeypatch.setattr("src.training.train.validate", mock_validate)
    
    train(cfg_path)
    
    # Read best.pt
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    best_pt = Path(cfg["experiment"]["model_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(best_pt)
    assert ckpt["epoch"] == 2

def test_checkpoint_selection_B(setup_configs, monkeypatch):
    """Sequence: 0.5, 0.4, 0.3 -> best epoch is 3"""
    cfg_path, _ = setup_configs
    
    val_sequence = [0.5, 0.4, 0.3]
    call_idx = 0
    def mock_validate(gen, loader, device, criterion_l1):
        nonlocal call_idx
        v = val_sequence[call_idx]
        call_idx += 1
        return v
        
    monkeypatch.setattr("src.training.train.validate", mock_validate)
    
    train(cfg_path)
    
    # Read best.pt
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    best_pt = Path(cfg["experiment"]["model_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(best_pt)
    assert ckpt["epoch"] == 3

def test_checkpoint_selection_C(setup_configs, monkeypatch):
    """train_one_run Sequence: 0.5, 0.3, 0.4 -> best epoch is 2"""
    cfg_path, cfg = setup_configs
    
    val_sequence = [0.5, 0.3, 0.4]
    call_idx = 0
    def mock_validate(gen, loader, device, criterion_l1):
        nonlocal call_idx
        v = val_sequence[call_idx]
        call_idx += 1
        return v
        
    monkeypatch.setattr("src.training.train.validate", mock_validate)
    
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():
        train_one_run(cfg)
    
    best_pt = Path(cfg["experiment"]["model_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(best_pt)
    assert ckpt["epoch"] == 2
