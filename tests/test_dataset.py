import csv
import numpy as np
import torch
from src.data.dataset import CloudDataset

def test_cloud_dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    csv_path = tmp_path / "split.csv"
    
    records = []
    for i in range(2):
        cloud = np.random.uniform(-1, 1, (256, 256, 3)).astype(np.float32)
        gt = np.random.uniform(-1, 1, (256, 256, 3)).astype(np.float32)
        
        c_path = data_dir / f"cloud_{i}.npy"
        g_path = data_dir / f"gt_{i}.npy"
        
        np.save(c_path, cloud)
        np.save(g_path, gt)
        
        records.append({
            "image_id": f"img_{i}",
            "patch_name": f"patch_{i}",
            "cloud_path": str(c_path),
            "gt_path": str(g_path)
        })
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "patch_name", "cloud_path", "gt_path"])
        writer.writeheader()
        writer.writerows(records)
        
    ds = CloudDataset(str(csv_path))
    assert len(ds) == 2
    
    for i in range(2):
        cloud_t, gt_t = ds[i]
        assert isinstance(cloud_t, torch.Tensor)
        assert isinstance(gt_t, torch.Tensor)
        assert cloud_t.shape == (3, 256, 256)
        assert gt_t.shape == (3, 256, 256)
        assert cloud_t.min() >= -1.0
        assert cloud_t.max() <= 1.0
        assert gt_t.min() >= -1.0
        assert gt_t.max() <= 1.0
