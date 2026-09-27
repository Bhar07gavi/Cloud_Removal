"""Pix2Pix GAN Training Loop with DagsHub MLflow & Checkpointing."""

import argparse
import json
import os
from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import CloudDataset
from src.models.discriminator import PatchGANDiscriminator
from src.models.generator import UNetGenerator
from src.training.losses import GANLoss, PixelLoss

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_sample_grid(generator: nn.Module, dataloader: DataLoader, device: torch.device, num_samples: int = 4):
    generator.eval()
    clouds, fakes, gts = [], [], []

    with torch.no_grad():
        for cloud, gt in dataloader:
            cloud = cloud.to(device)
            fake = generator(cloud)
            clouds.append(cloud.cpu())
            fakes.append(fake.cpu())
            gts.append(gt.cpu())
            if sum(c.shape[0] for c in clouds) >= num_samples:
                break

    clouds = torch.cat(clouds)[:num_samples]
    fakes = torch.cat(fakes)[:num_samples]
    gts = torch.cat(gts)[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        for j, (img, title) in enumerate([(clouds[i], "Cloudy"), (fakes[i], "Generated"), (gts[i], "Ground Truth")]):
            ax = axes[i, j]
            # Denormalize [-1, 1] -> [0, 1]
            img_np = ((img + 1.0) / 2.0).permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img_np)
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    generator.train()
    return fig

def train(config_path: str, epochs_override: int | None = None):
    cfg = load_config(config_path)
    ds_cfg = load_config(cfg["dataset"]["config_path"])

    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    device_str = cfg["training"].get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu")
    epochs = epochs_override if epochs_override is not None else cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["optimizer"]["lr"]
    betas = tuple(cfg["optimizer"]["betas"])
    lambda_l1 = cfg["loss"]["lambda_l1"]

    checkpoint_dir = Path(cfg["experiment"]["model_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = ds_cfg["name"]
    splits_dir = Path(ds_cfg["paths"].get("splits_dir", "data/splits"))

    train_ds = CloudDataset(str(splits_dir / f"{dataset_name}_train.csv"))
    val_ds = CloudDataset(str(splits_dir / f"{dataset_name}_val.csv"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    bands = ds_cfg["bands"]["count"]
    gen = UNetGenerator(in_channels=bands, out_channels=bands).to(device)
    disc = PatchGANDiscriminator(in_channels=bands * 2).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    criterion_adv = GANLoss(cfg["loss"].get("adversarial", "bce")).to(device)
    criterion_l1 = PixelLoss().to(device)

    # MLflow Setup
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    if tracking_uri and "<YOUR_" not in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "cloud-removal"))

    best_val_loss = float("inf")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_l1": lambda_l1,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        })

        for epoch in range(1, epochs + 1):
            gen.train()
            disc.train()
            g_loss_acc, d_loss_acc, l1_loss_acc = 0.0, 0.0, 0.0
            batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for cloud, gt in pbar:
                cloud, gt = cloud.to(device), gt.to(device)

                # --- 1. Train Discriminator ---
                fake = gen(cloud)
                pred_real = disc(torch.cat([cloud, gt], dim=1))
                loss_d_real = criterion_adv(pred_real, is_real=True)

                pred_fake = disc(torch.cat([cloud, fake.detach()], dim=1))
                loss_d_fake = criterion_adv(pred_fake, is_real=False)

                loss_d = (loss_d_real + loss_d_fake) * 0.5
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # --- 2. Train Generator ---
                pred_fake = disc(torch.cat([cloud, fake], dim=1))
                loss_g_adv = criterion_adv(pred_fake, is_real=True)
                loss_g_l1 = criterion_l1(fake, gt)
                loss_g = loss_g_adv + lambda_l1 * loss_g_l1

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                g_loss_acc += loss_g.item()
                d_loss_acc += loss_d.item()
                l1_loss_acc += loss_g_l1.item()
                batches += 1

                pbar.set_postfix({"G": f"{loss_g.item():.3f}", "D": f"{loss_d.item():.3f}", "L1": f"{loss_g_l1.item():.3f}"})

            avg_g = g_loss_acc / max(batches, 1)
            avg_d = d_loss_acc / max(batches, 1)
            avg_l1 = l1_loss_acc / max(batches, 1)

            print(f"Epoch {epoch}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f} | L1: {avg_l1:.4f}")

            mlflow.log_metrics({"train/G_loss": avg_g, "train/D_loss": avg_d, "train/L1_loss": avg_l1}, step=epoch)

            # Sample Grid
            if epoch % cfg["training"].get("sample_grid_every_n_epochs", 5) == 0:
                fig = make_sample_grid(gen, val_loader, device)
                grid_file = checkpoint_dir / f"grid_epoch_{epoch:03d}.png"
                fig.savefig(str(grid_file), dpi=100)
                plt.close(fig)
                mlflow.log_artifact(str(grid_file), "sample_grids")

            # Checkpoint
            if avg_l1 < best_val_loss:
                best_val_loss = avg_l1
                torch.save({"epoch": epoch, "generator": gen.state_dict(), "discriminator": disc.state_dict()}, str(checkpoint_dir / "best.pt"))
                print(f"  * New best model saved (val L1: {best_val_loss:.4f})")

        print(f"Training Complete! Best L1: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Callable entry-point for Optuna (and any other programmatic caller)
# ---------------------------------------------------------------------------

def train_one_run(config: dict) -> float:
    """Train for config["training"]["epochs"] epochs and return mean val SSIM.

    MLflow-run-neutral: logs metrics to whatever run is currently active (the
    nested run opened by tune.py) but does NOT call mlflow.start_run() itself.
    If no active run exists the log calls are silently no-ops.

    Parameters
    ----------
    config : dict
        Fully-merged config dict with the same schema as train_rice.yaml.
        Callers (e.g. tune.py) build this by loading the base YAML and
        overriding keys they want to sweep (lr, batch_size, lambda_l1, epochs,
        model_dir).

    Returns
    -------
    float
        Mean SSIM over the validation set after training completes.
    """
    cfg = config
    with open(cfg["dataset"]["config_path"], "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    device_str = cfg["training"].get("device", "auto")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_str in ("auto", "cuda") else "cpu"
    )
    epochs     = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr         = cfg["optimizer"]["lr"]
    betas      = tuple(cfg["optimizer"]["betas"])
    lambda_l1  = cfg["loss"]["lambda_l1"]

    checkpoint_dir = Path(cfg["experiment"]["model_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = ds_cfg["name"]
    splits_dir   = Path(ds_cfg["paths"].get("splits_dir", "data/splits"))

    train_ds     = CloudDataset(str(splits_dir / f"{dataset_name}_train.csv"))
    val_ds       = CloudDataset(str(splits_dir / f"{dataset_name}_val.csv"))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    bands = ds_cfg["bands"]["count"]
    gen   = UNetGenerator(in_channels=bands, out_channels=bands).to(device)
    disc  = PatchGANDiscriminator(in_channels=bands * 2).to(device)

    opt_g = torch.optim.Adam(gen.parameters(),  lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    criterion_adv = GANLoss(cfg["loss"].get("adversarial", "bce")).to(device)
    criterion_l1  = PixelLoss().to(device)

    best_val_l1 = float("inf")

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        g_acc = d_acc = l1_acc = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"  [trial] Epoch {epoch}/{epochs}", leave=False)
        for cloud, gt in pbar:
            cloud, gt = cloud.to(device), gt.to(device)

            # Train Discriminator
            fake      = gen(cloud)
            pred_real = disc(torch.cat([cloud, gt],            dim=1))
            pred_fake = disc(torch.cat([cloud, fake.detach()], dim=1))
            loss_d    = (criterion_adv(pred_real, is_real=True) +
                         criterion_adv(pred_fake, is_real=False)) * 0.5
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train Generator
            pred_fake  = disc(torch.cat([cloud, fake], dim=1))
            loss_g_adv = criterion_adv(pred_fake, is_real=True)
            loss_g_l1  = criterion_l1(fake, gt)
            loss_g     = loss_g_adv + lambda_l1 * loss_g_l1
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_acc   += loss_g.item()
            d_acc   += loss_d.item()
            l1_acc  += loss_g_l1.item()
            batches += 1
            pbar.set_postfix({
                "G": f"{loss_g.item():.3f}",
                "D": f"{loss_d.item():.3f}",
                "L1": f"{loss_g_l1.item():.3f}",
            })

        avg_g   = g_acc  / max(batches, 1)
        avg_d   = d_acc  / max(batches, 1)
        avg_l1  = l1_acc / max(batches, 1)
        print(f"  [trial] Epoch {epoch}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f} | L1: {avg_l1:.4f}")

        # Log to the active nested MLflow run (if any)
        try:
            mlflow.log_metrics(
                {"trial/G_loss": avg_g, "trial/D_loss": avg_d, "trial/L1_loss": avg_l1},
                step=epoch,
            )
        except Exception:
            pass  # no active run → ignore

        if avg_l1 < best_val_l1:
            best_val_l1 = avg_l1
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                },
                str(checkpoint_dir / "best.pt"),
            )

    # ---- Compute SSIM on val set using the best saved checkpoint ----
    ckpt = torch.load(str(checkpoint_dir / "best.pt"), map_location=device)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()

    ssim_scores: list[float] = []
    with torch.no_grad():
        for cloud, gt in val_loader:
            cloud = cloud.to(device)
            fake  = gen(cloud)
            for i in range(fake.shape[0]):
                fake_np = (
                    (fake[i] + 1.0) / 2.0
                ).permute(1, 2, 0).cpu().numpy().clip(0, 1).astype("float32")
                gt_np = (
                    (gt[i] + 1.0) / 2.0
                ).permute(1, 2, 0).cpu().numpy().clip(0, 1).astype("float32")
                ssim_scores.append(
                    float(structural_similarity(gt_np, fake_np, data_range=1.0, channel_axis=2))
                )

    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    print(f"  [trial] Val mean SSIM = {mean_ssim:.6f}")
    return mean_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_rice.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.epochs)
