"""Preprocess RICE1 and RICE2 datasets into tiled 256x256 normalized patches."""

import argparse
import csv
from pathlib import Path
import random
import numpy as np
from PIL import Image
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def tile_image(img_array: np.ndarray, patch_size: int) -> list[tuple[np.ndarray, int, int]]:
    h, w = img_array.shape[:2]
    patches = []
    for r_idx, r in enumerate(range(0, h - patch_size + 1, patch_size)):
        for c_idx, c in enumerate(range(0, w - patch_size + 1, patch_size)):
            patch = img_array[r : r + patch_size, c : c + patch_size]
            patches.append((patch, r_idx, c_idx))
    return patches

def normalize_patch(patch: np.ndarray, cfg_norm: dict) -> np.ndarray:
    patch = patch.astype(np.float32)
    # [0, 255] -> [-1.0, 1.0]
    return (patch / 127.5) - 1.0

def discover_rice_pairs(raw_dir: str) -> list[tuple[str, str, str]]:
    """Discovers pairs from raw_dir, automatically handling flat layouts
    or subdirectories like raw_dir/rice1 and raw_dir/RICE2.
    """
    raw = Path(raw_dir)
    subdirs = [d for d in raw.iterdir() if d.is_dir() and (d / "cloud").is_dir()]
    if not subdirs and (raw / "cloud").is_dir():
        subdirs = [raw]

    if not subdirs:
        raise FileNotFoundError(f"No valid image subdirectories found under {raw_dir}")

    all_pairs = []
    for base in subdirs:
        cloud_dir = base / "cloud"
        gt_dir = None
        for candidate_name in ["label", "ground_truth", "gt"]:
            if (base / candidate_name).is_dir():
                gt_dir = base / candidate_name
                break

        if gt_dir is None:
            print(f"Warning: No ground truth directory found for {base.name}, skipping.")
            continue

        cloud_files = sorted(cloud_dir.glob("*.*"))
        for cf in cloud_files:
            gt_candidates = list(gt_dir.glob(cf.stem + ".*"))
            if gt_candidates:
                prefix = f"{base.name}_" if len(subdirs) > 1 else ""
                all_pairs.append((f"{prefix}{cf.stem}", str(cf), str(gt_candidates[0])))

    print(f"Discovered total {len(all_pairs)} image pairs from {len(subdirs)} source(s).")
    return all_pairs

def split_image_ids(image_ids: list[str], train_frac: float, val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

def preprocess(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["name"]
    patch_size = cfg["patch_size"]
    raw_dir = cfg["paths"]["raw"]
    processed_dir = cfg["paths"]["processed"]
    manifest_path = cfg["paths"]["pairs_manifest"]
    splits_dir = Path(cfg["paths"].get("splits_dir", "data/splits"))

    pairs = discover_rice_pairs(raw_dir)
    image_ids = [p[0] for p in pairs]
    pairs_dict = {p[0]: (p[1], p[2]) for p in pairs}

    splits = split_image_ids(
        image_ids,
        cfg["split"]["train"],
        cfg["split"]["val"],
        cfg["split"]["test"],
        cfg["split"]["seed"],
    )

    cloud_out = Path(processed_dir) / "cloud"
    gt_out = Path(processed_dir) / "gt"
    for d in [cloud_out, gt_out, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_records = []
    for img_id, (cloud_path, gt_path) in pairs_dict.items():
        try:
            cloud_img = np.array(Image.open(cloud_path).convert("RGB"))
            gt_img = np.array(Image.open(gt_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
            continue

        cloud_patches = tile_image(cloud_img, patch_size)
        gt_patches = tile_image(gt_img, patch_size)

        for (c_patch, r, c), (g_patch, _, _) in zip(cloud_patches, gt_patches):
            c_norm = normalize_patch(c_patch, cfg["normalization"])
            g_norm = normalize_patch(g_patch, cfg["normalization"])

            patch_name = f"{img_id}_r{r}_c{c}"
            c_path = cloud_out / f"{patch_name}.npy"
            g_path = gt_out / f"{patch_name}.npy"

            np.save(str(c_path), c_norm)
            np.save(str(g_path), g_norm)
            all_records.append((img_id, patch_name, str(c_path).replace('\\', '/'), str(g_path).replace('\\', '/')))

    print(f"Total patches generated: {len(all_records)}")

    # Write manifest
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "patch_name", "cloud_path", "gt_path"])
        writer.writerows(all_records)

    # Write splits
    for split_name, split_ids in splits.items():
        split_set = set(split_ids)
        split_records = [r for r in all_records if r[0] in split_set]
        split_csv = splits_dir / f"{dataset_name}_{split_name}.csv"
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "patch_name", "cloud_path", "gt_path"])
            writer.writerows(split_records)
        print(f"Saved {split_name} split ({len(split_records)} patches) -> {split_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_rice.yaml")
    args = parser.parse_args()
    preprocess(args.config)
