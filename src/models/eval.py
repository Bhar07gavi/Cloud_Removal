"""Evaluate the trained pix2pix GAN on the test split.

Computes per-image and mean PSNR / SSIM using scikit-image, then logs
summary metrics to the existing MLflow experiment on DAGsHub.

Usage (from repo root, same CWD convention as train.py):
    python -m src.models.eval --config configs/train_rice.yaml
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import CloudDataset
from src.models.generator import UNetGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Identical to the loader used in train.py."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output from [-1, 1] (Tanh) back to [0, 1] float32.

    Mirrors the inverse of normalize_patch() in src/data/preprocess.py:
        patch = (raw_uint8 / 127.5) - 1.0
    So the inverse is:
        uint8_approx = (tensor + 1.0) / 2.0  ->  [0, 1]

    Consistent with the denorm used in make_sample_grid() in train.py (line 61).
    """
    arr = ((tensor + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
    return arr.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(config_path: str) -> None:
    """Run full test-set evaluation and log metrics to MLflow."""
    cfg = load_config(config_path)
    ds_cfg = load_config(cfg["dataset"]["config_path"])

    # ---- Device ----
    device_str = cfg["training"].get("device", "auto")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_str in ("auto", "cuda") else "cpu"
    )
    print(f"Using device: {device}")

    # ---- Load test dataset (same CloudDataset used in training) ----
    dataset_name = ds_cfg["name"]                                  # "rice_combined"
    splits_dir   = Path(ds_cfg["paths"].get("splits_dir", "data/splits"))
    test_csv     = splits_dir / f"{dataset_name}_test.csv"

    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test split CSV not found: {test_csv}\n"
            "Run preprocessing first: "
            "python -m src.data.preprocess --config configs/dataset_rice.yaml"
        )

    test_ds     = CloudDataset(str(test_csv))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} patches  ({test_csv})")

    # ---- Load generator weights ----
    bands           = ds_cfg["bands"]["count"]                     # 3 (RGB)
    checkpoint_path = Path(cfg["experiment"]["model_dir"]) / "checkpoints" / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first: python src/training/train.py --config configs/train_rice.yaml"
        )

    gen = UNetGenerator(in_channels=bands, out_channels=bands).to(device)

    # Checkpoint format saved by train.py:
    #   {"epoch": int, "generator": state_dict, "discriminator": state_dict}
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    gen.load_state_dict(checkpoint["generator"])
    gen.eval()
    print(f"Loaded generator from {checkpoint_path}  (epoch {checkpoint.get('epoch', '?')})")

    # ---- MLflow setup (mirrors train.py exactly) ----
    mlflow_cfg   = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    if tracking_uri and "<YOUR_" not in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "cloud-removal"))

    # ---- Inference + metrics ----
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []

    header = f"{'Patch':>6}  {'PSNR (dB)':>10}  {'SSIM':>8}"
    print("\n" + header)
    print("-" * len(header))

    with torch.no_grad():
        for idx, (cloud, gt) in enumerate(tqdm(test_loader, desc="Evaluating", leave=True)):
            cloud = cloud.to(device)    # (1, C, H, W)  in [-1, 1]
            fake  = gen(cloud)          # (1, C, H, W)  in [-1, 1]  (Tanh)

            # Denormalize both to [0, 1] float32 HWC
            fake_np = denormalize(fake.squeeze(0))   # (H, W, C)
            gt_np   = denormalize(gt.squeeze(0))     # (H, W, C)

            # PSNR - data_range=1.0 because images are in [0, 1]
            psnr = peak_signal_noise_ratio(gt_np, fake_np, data_range=1.0)

            # SSIM - channel_axis=2 for HWC multi-channel, data_range=1.0
            ssim = structural_similarity(
                gt_np, fake_np,
                data_range=1.0,
                channel_axis=2,
            )

            psnr_scores.append(float(psnr))
            ssim_scores.append(float(ssim))

            print(f"{idx + 1:>6}  {psnr:>10.4f}  {ssim:>8.6f}")

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))

    print("\n" + "=" * len(header))
    print(f"{'Mean':>6}  {mean_psnr:>10.4f}  {mean_ssim:>8.6f}")
    print(f"\nTotal patches evaluated: {len(psnr_scores)}")

    # ---- Log to MLflow under a dedicated 'evaluation' run ----
    with mlflow.start_run(run_name="evaluation", tags={"stage": "evaluation"}):
        mlflow.log_params({
            "checkpoint":   str(checkpoint_path),
            "test_csv":     str(test_csv),
            "test_patches": len(psnr_scores),
            "device":       str(device),
        })
        mlflow.log_metrics({
            "eval/mean_psnr": mean_psnr,
            "eval/mean_ssim": mean_ssim,
        })

        # Log per-patch metrics as a step series so they appear as line charts
        for step, (psnr, ssim) in enumerate(zip(psnr_scores, ssim_scores)):
            mlflow.log_metrics(
                {"eval/psnr_per_patch": psnr, "eval/ssim_per_patch": ssim},
                step=step,
            )

    print(
        f"\nMLflow run logged under experiment "
        f"'{mlflow_cfg.get('experiment_name', 'cloud-removal')}' -> tag: evaluation\n"
        f"  eval/mean_psnr = {mean_psnr:.4f} dB\n"
        f"  eval/mean_ssim = {mean_ssim:.6f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained pix2pix GAN on test split (PSNR + SSIM)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_rice.yaml",
        help="Path to training config YAML (same one used with train.py).",
    )
    args = parser.parse_args()
    evaluate(args.config)
