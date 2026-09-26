"""Preprocessing pipeline: discovers RICE1/RICE2 pairs, tiles to 256x256,
normalizes to [-1,1], and creates image-ID-level train/val/test splits.

Usage:
    python -m src.data.preprocess --config configs/dataset.yaml
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


def discover_pairs(dataset_dir: str) -> list[tuple[str, str, str]]:
    """Discover matched (cloudy, clean) image pairs from a RICE directory.

    Args:
        dataset_dir: Path to a RICE dataset root (must contain cloud/ and label/ subdirs).

    Returns:
        Sorted list of (image_id, cloud_path, label_path) tuples.
        image_id is formatted as '{dataset_name}_{stem}', e.g. 'RICE1_42'.
    """
    dataset_path = Path(dataset_dir)
    parent_name = dataset_path.name
    cloud_dir = dataset_path / "cloud"
    label_dir = dataset_path / "label"

    if not cloud_dir.exists() or not label_dir.exists():
        print(f"[WARN] Skipping {dataset_dir}: cloud/ or label/ not found.")
        return []

    pairs: list[tuple[str, str, str]] = []
    for cloud_file in sorted(cloud_dir.glob("*.png")):
        label_file = label_dir / cloud_file.name
        if label_file.exists():
            image_id = f"{parent_name}_{cloud_file.stem}"
            pairs.append((image_id, str(cloud_file), str(label_file)))

    return pairs


def tile_and_normalize(image_path: str, patch_size: int = 256) -> list[np.ndarray]:
    """Tile an image into non-overlapping patches and normalize to [-1, 1].

    Args:
        image_path: Path to the source image.
        patch_size: Side length of square patches.

    Returns:
        List of normalized patches, each shape (patch_size, patch_size, 3) float32.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    patches: list[np.ndarray] = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = arr[i : i + patch_size, j : j + patch_size, :]
            # Normalize [0, 255] → [-1.0, 1.0]
            patch_norm = (patch.astype(np.float32) / 127.5) - 1.0
            patches.append(patch_norm)

    return patches


def split_image_ids(
    image_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split image IDs into train/val/test sets (image-level, no leakage).

    Args:
        image_ids: Full list of image IDs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng = random.Random(seed)
    shuffled = image_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def run_preprocessing(config_path: str) -> None:
    """Execute the full preprocessing pipeline using the given config.

    Reads RICE1/RICE2 directories, tiles images into 256×256 patches,
    normalizes pixel values to [-1, 1], saves as .npy, and writes CSV
    manifests for each split.

    Args:
        config_path: Path to the dataset YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parse nested config structure
    raw_cfg = config.get("raw_data", {})
    prep_cfg = config.get("preprocess", {})
    split_cfg = config.get("split", {})

    rice1_dir = raw_cfg.get("rice1_dir", "dataset/RICE1")
    rice2_dir = raw_cfg.get("rice2_dir", "dataset/RICE2")
    patch_size = prep_cfg.get("patch_size", 256)
    output_dir = Path(prep_cfg.get("output_dir", "data/processed"))
    splits_dir = Path(prep_cfg.get("splits_dir", "data/splits"))

    train_ratio = split_cfg.get("train", 0.8)
    val_ratio = split_cfg.get("val", 0.1)
    seed = split_cfg.get("seed", 42)

    # --- Discover image pairs ---
    print(f"Discovering pairs from {rice1_dir} and {rice2_dir} ...")
    pairs_rice1 = discover_pairs(rice1_dir)
    pairs_rice2 = discover_pairs(rice2_dir)
    all_pairs = pairs_rice1 + pairs_rice2
    print(f"  RICE1: {len(pairs_rice1)} pairs | RICE2: {len(pairs_rice2)} pairs | Total: {len(all_pairs)}")

    pair_dict = {img_id: (cloud, label) for img_id, cloud, label in all_pairs}
    all_image_ids = sorted(pair_dict.keys())

    # --- Split at image-ID level ---
    train_ids, val_ids, test_ids = split_image_ids(
        all_image_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    print(f"  Split (image-level): train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # --- Create output directories ---
    cloud_out = output_dir / "cloud"
    label_out = output_dir / "label"
    cloud_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # --- Tile, normalize, save, and write manifests ---
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    patch_counts: dict[str, int] = {}

    for split_name, ids in split_map.items():
        csv_path = splits_dir / f"{split_name}.csv"
        count = 0

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["cloud_path", "gt_path"])

            for img_id in ids:
                cloud_path, label_path = pair_dict[img_id]
                cloud_patches = tile_and_normalize(cloud_path, patch_size)
                label_patches = tile_and_normalize(label_path, patch_size)

                for i, (c_patch, l_patch) in enumerate(zip(cloud_patches, label_patches)):
                    fname = f"{img_id}_patch{i}.npy"
                    c_out = cloud_out / fname
                    l_out = label_out / fname

                    np.save(c_out, c_patch)
                    np.save(l_out, l_patch)

                    # Store relative paths (forward slashes for cross-platform CSV)
                    rel_c = str(c_out).replace("\\", "/")
                    rel_l = str(l_out).replace("\\", "/")
                    writer.writerow([rel_c, rel_l])
                    count += 1

        patch_counts[split_name] = count
        print(f"  {split_name}: {count} patches -> {csv_path}")

    total_patches = sum(patch_counts.values())
    print(f"\nDone! {len(all_pairs)} images -> {total_patches} patches total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RICE datasets for cloud removal training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to the dataset YAML config file.",
    )
    args = parser.parse_args()
    run_preprocessing(args.config)
