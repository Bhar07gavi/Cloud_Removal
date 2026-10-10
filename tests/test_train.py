import pytest
import numpy as np
import yaml
import torch
import csv
from pathlib import Path

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
            "tracking_uri": Path(tmp_path / "mlruns").as_uri(),
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

def test_mlflow_generator_logging(setup_configs, monkeypatch):
    """Assert generator model and traceability are logged to MLflow in train(), but not in train_one_run()."""
    cfg_path, cfg = setup_configs
    
    # 2 epochs: mock validate so best epoch is 1 (0.2, 0.4)
    val_sequence = [0.2, 0.4]
    call_idx = 0
    def mock_validate(gen, loader, device, criterion_l1):
        nonlocal call_idx
        v = val_sequence[call_idx]
        call_idx += 1
        return v
        
    monkeypatch.setattr("src.training.train.validate", mock_validate)
    
    model_info = train(cfg_path, epochs_override=2)
    
    import mlflow
    import mlflow.pytorch
    from src.models.generator import UNetGenerator
    
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    runs = client.search_runs(exp.experiment_id)
    assert len(runs) > 0
    train_run = runs[0]
    
    # a) assert params dataset_md5 and git_commit exist and metrics best_epoch and best_val_l1 exist
    assert "dataset_md5" in train_run.data.params
    assert "git_commit" in train_run.data.params
    assert "best_epoch" in train_run.data.metrics
    assert "best_val_l1" in train_run.data.metrics
    assert train_run.data.metrics["best_epoch"] == 1
    
    # b) assert artifact list contains "generator" model and "configs" folder
    artifacts = [a.path for a in client.list_artifacts(train_run.info.run_id)]
    assert "configs" in artifacts
    if hasattr(client, "search_logged_models"):
        logged_models = [m.name for m in client.search_logged_models(experiment_ids=[exp.experiment_id])]
        assert "generator" in artifacts or "generator" in logged_models
    else:
        assert "generator" in artifacts
    
    # c) load logged model with mlflow.pytorch.load_model on a fixed random input and assert torch.allclose against generator rebuilt from best.pt, using eval mode in both
    if model_info is not None and hasattr(model_info, "model_uri"):
        model_uri = model_info.model_uri
    elif hasattr(client, "search_logged_models"):
        matching = [m for m in client.search_logged_models(experiment_ids=[exp.experiment_id]) if getattr(m, "source_run_id", None) == train_run.info.run_id]
        model_uri = matching[0].model_uri if matching else f"runs:/{train_run.info.run_id}/generator"
    else:
        model_uri = f"runs:/{train_run.info.run_id}/generator"
    logged_model = mlflow.pytorch.load_model(model_uri)
    logged_model.eval()
    
    best_pt = Path(cfg["experiment"]["model_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(best_pt, map_location="cpu")
    rebuilt_gen = UNetGenerator(3, 3)
    rebuilt_gen.load_state_dict(ckpt["generator"])
    rebuilt_gen.eval()
    
    torch.manual_seed(123)
    test_inp = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out_logged = logged_model(test_inp)
        out_rebuilt = rebuilt_gen(test_inp)
    assert torch.allclose(out_logged, out_rebuilt)
    
    # d) assert a train_one_run() run produces NO "generator" model artifact
    cfg["training"]["epochs"] = 2
    val_sequence = [0.2, 0.4]
    call_idx = 0
    with mlflow.start_run() as trial_run:
        train_one_run(cfg)
        
    trial_artifacts = [a.path for a in client.list_artifacts(trial_run.info.run_id)]
    assert "generator" not in trial_artifacts
    if hasattr(client, "search_logged_models"):
        trial_models = [m.name for m in client.search_logged_models(experiment_ids=[exp.experiment_id]) if getattr(m, "source_run_id", None) == trial_run.info.run_id]
        assert "generator" not in trial_models

