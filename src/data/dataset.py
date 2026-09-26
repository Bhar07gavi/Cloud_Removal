"""PyTorch Dataset for paired satellite patches."""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class CloudDataset(Dataset):
    """Loads cloudy and cloud-free ground-truth .npy patches."""

    def __init__(self, split_csv: str):
        self.records = []
        with open(split_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((row["cloud_path"], row["gt_path"]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cloud_path, gt_path = self.records[idx]

        # Load float32 (H, W, C) arrays
        cloud = np.load(cloud_path)
        gt = np.load(gt_path)

        # Transpose to (C, H, W) for PyTorch
        cloud = np.transpose(cloud, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        return torch.from_numpy(cloud), torch.from_numpy(gt)
