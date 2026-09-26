"""Full training loop for Pix2Pix cloud removal GAN.

Supports config-driven training, periodic checkpointing, validation image
logging, and (optional) DagsHub/MLflow experiment tracking.

Usage:
    python -m src.training.train --config configs/train.yaml --epochs 50
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.discriminator import build_discriminator
from src.models.generator import build_generator
from src.training.losses import build_gan_loss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor from [-1, 1] to [0, 1] numpy array for visualization.

    Args:
        tensor: Image tensor of shape (C, H, W) in [-1, 1].

    Returns:
        Numpy array of shape (H, W, C) in [0, 1], clipped.
    """
    img = tensor.detach().cpu().numpy()
    img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
    img = np.clip(img, 0.0, 1.0)
    return img.transpose(1, 2, 0)  # CHW → HWC


def make_sample_grid(
    cloudy: torch.Tensor,
    generated: torch.Tensor,
    ground_truth: torch.Tensor,
    n_samples: int = 4,
) -> plt.Figure:
    """Create a matplotlib figure with columns: Cloudy | Generated | Ground Truth.

    Args:
        cloudy: Batch of cloudy inputs (N, C, H, W).
        generated: Batch of generated outputs (N, C, H, W).
        ground_truth: Batch of ground truth images (N, C, H, W).
        n_samples: Number of rows to display.

    Returns:
        Matplotlib Figure object.
    """
    n = min(n_samples, cloudy.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    titles = ["Cloudy Input", "Generated", "Ground Truth"]
    for i in range(n):
        images = [
            denormalize(cloudy[i]),
            denormalize(generated[i]),
            denormalize(ground_truth[i]),
        ]
        for j, (ax, img) in enumerate(zip(axes[i], images)):
            ax.imshow(img)
            ax.set_title(titles[j], fontsize=10)
            ax.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str, epochs_override: int | None = None) -> None:
    """Execute the full GAN training loop.

    Args:
        config_path: Path to the training YAML config.
        epochs_override: If provided, overrides the epoch count in config.
    """
    # --- Load config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpoint", {})
    data_cfg = config.get("data", {})
    mlflow_cfg = config.get("mlflow", {})

    batch_size = train_cfg.get("batch_size", 16)
    epochs = epochs_override if epochs_override is not None else train_cfg.get("epochs", 50)
    lr = train_cfg.get("lr", 0.0002)
    beta1 = train_cfg.get("beta1", 0.5)
    beta2 = train_cfg.get("beta2", 0.999)
    lambda_l1 = train_cfg.get("lambda_l1", 100)
    num_workers = train_cfg.get("num_workers", 2)

    save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
    save_every = ckpt_cfg.get("save_every", 5)
    log_images_every = ckpt_cfg.get("log_images_every", 5)

    splits_dir = data_cfg.get("splits_dir", "data/splits")
    sample_images_dir = save_dir / "sample_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_images_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading data from {splits_dir} ...")
    loaders = build_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Models ---
    generator = build_generator().to(device)
    discriminator = build_discriminator().to(device)
    print(f"  Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # --- Optimizers ---
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # --- Loss ---
    criterion = build_gan_loss(lambda_l1=lambda_l1)

    # --- Optional MLflow tracking ---
    mlflow_enabled = False
    try:
        tracking_uri = mlflow_cfg.get("tracking_uri", "")
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "cloud-removal-gan"))
            mlflow_enabled = True
            print(f"  MLflow tracking: {tracking_uri}")
    except Exception as e:
        print(f"  MLflow not available ({e}), proceeding without tracking.")

    # --- Training loop ---
    best_val_l1 = float("inf")

    if mlflow_enabled:
        import mlflow
        mlflow.start_run()
        mlflow.log_params({
            "batch_size": batch_size, "epochs": epochs, "lr": lr,
            "beta1": beta1, "beta2": beta2, "lambda_l1": lambda_l1,
        })

    try:
        for epoch in range(1, epochs + 1):
            generator.train()
            discriminator.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_l1_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for cloudy, gt in pbar:
                cloudy = cloudy.to(device)
                gt = gt.to(device)

                # --- Train Discriminator ---
                generated = generator(cloudy)

                pred_real = discriminator(cloudy, gt)
                pred_fake = discriminator(cloudy, generated.detach())
                d_loss = criterion.discriminator_loss(pred_real, pred_fake)

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # --- Train Generator ---
                pred_fake_for_g = discriminator(cloudy, generated)
                g_loss, adv_loss, l1_loss = criterion.generator_loss(
                    pred_fake_for_g, generated, gt
                )

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_l1_loss += l1_loss.item()
                n_batches += 1

                pbar.set_postfix({
                    "G": f"{g_loss.item():.4f}",
                    "D": f"{d_loss.item():.4f}",
                    "L1": f"{l1_loss.item():.4f}",
                })

            # --- Epoch averages ---
            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            avg_l1 = epoch_l1_loss / max(n_batches, 1)
            print(f"Epoch {epoch}/{epochs} — G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | L1_loss: {avg_l1:.4f}")

            # --- Log to MLflow ---
            if mlflow_enabled:
                import mlflow
                mlflow.log_metrics({
                    "train/G_loss": avg_g,
                    "train/D_loss": avg_d,
                    "train/L1_loss": avg_l1,
                }, step=epoch)

            # --- Validation L1 ---
            generator.eval()
            val_l1_total = 0.0
            val_batches = 0
            val_cloudy_sample = None
            val_gt_sample = None
            val_gen_sample = None

            with torch.no_grad():
                for cloudy_v, gt_v in val_loader:
                    cloudy_v = cloudy_v.to(device)
                    gt_v = gt_v.to(device)
                    gen_v = generator(cloudy_v)
                    val_l1_total += nn.functional.l1_loss(gen_v, gt_v).item()
                    val_batches += 1

                    if val_cloudy_sample is None:
                        val_cloudy_sample = cloudy_v
                        val_gt_sample = gt_v
                        val_gen_sample = gen_v

            avg_val_l1 = val_l1_total / max(val_batches, 1)
            print(f"  Val L1: {avg_val_l1:.4f}")

            if mlflow_enabled:
                import mlflow
                mlflow.log_metric("val/L1_loss", avg_val_l1, step=epoch)

            # --- Sample image grid ---
            if epoch % log_images_every == 0 and val_cloudy_sample is not None:
                fig = make_sample_grid(val_cloudy_sample, val_gen_sample, val_gt_sample)
                fig_path = sample_images_dir / f"epoch_{epoch:03d}.png"
                fig.savefig(fig_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved sample grid -> {fig_path}")

                if mlflow_enabled:
                    import mlflow
                    mlflow.log_artifact(str(fig_path), artifact_path="sample_images")

            # --- Checkpointing ---
            if epoch % save_every == 0:
                ckpt = {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": opt_g.state_dict(),
                    "optimizer_d_state_dict": opt_d.state_dict(),
                    "val_l1": avg_val_l1,
                }
                ckpt_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Saved checkpoint -> {ckpt_path}")

            # --- Best model ---
            if avg_val_l1 < best_val_l1:
                best_val_l1 = avg_val_l1
                best_ckpt = {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": opt_g.state_dict(),
                    "optimizer_d_state_dict": opt_d.state_dict(),
                    "val_l1": avg_val_l1,
                }
                best_path = save_dir / "best.pt"
                torch.save(best_ckpt, best_path)
                print(f"  * New best model (val L1={avg_val_l1:.4f}) -> {best_path}")

    finally:
        if mlflow_enabled:
            import mlflow
            mlflow.end_run()

    print(f"\nTraining complete. Best val L1: {best_val_l1:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pix2Pix cloud removal GAN")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count from config.",
    )
    args = parser.parse_args()
    train(args.config, epochs_override=args.epochs)
