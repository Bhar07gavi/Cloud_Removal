"""Optuna hyperparameter sweep for the Pix2Pix cloud-removal model.

Each trial:
  - Samples lr, batch_size, lambda_l1 from the defined search space
  - Runs a short training trial (default 5 epochs) via train_one_run()
  - Returns val-set mean SSIM as the objective to **maximise**
  - Is logged as a nested MLflow run under a parent 'optuna-sweep' run

Usage (from repo root):
    python -m src.training.tune \\
        --config configs/train_rice.yaml \\
        --n-trials 20 \\
        --trial-epochs 5

Requirements:
    pip install optuna  (added to requirements.txt)

Baseline reference: 20-epoch run -> mean SSIM 0.8281
"""

import argparse
import copy
import sys
import traceback
from pathlib import Path

import mlflow
import optuna
import yaml

from src.training.train import load_config, train_one_run

# ---------------------------------------------------------------------------
# Silence Optuna's verbose per-trial logging -- we handle it ourselves
# ---------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(base_cfg: dict, trial_epochs: int, trial_dir_root: Path):
    """Factory that closes over the base config and returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        # ---- 1. Sample hyperparameters ----
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        lambda_l1 = trial.suggest_float("lambda_l1", 50.0, 150.0)

        print(
            f"\n{'='*60}\n"
            f"Trial #{trial.number:>3} | lr={lr:.2e}  bs={batch_size}  lambda_l1={lambda_l1:.1f}\n"
            f"{'='*60}"
        )

        # ---- 2. Build per-trial config (deep copy so trials don't share state) ----
        cfg = copy.deepcopy(base_cfg)
        cfg["optimizer"]["lr"] = float(lr)
        cfg["training"]["batch_size"] = int(batch_size)
        cfg["training"]["epochs"] = trial_epochs
        cfg["loss"]["lambda_l1"] = float(lambda_l1)

        # Each trial writes its checkpoint to an isolated directory so trials
        # never overwrite each other's best.pt
        trial_model_dir = trial_dir_root / f"trial_{trial.number:03d}"
        cfg["experiment"]["model_dir"] = str(trial_model_dir)

        # ---- 3. Open nested MLflow run for this trial ----
        run_name = f"optuna_trial_{trial.number:03d}"
        with mlflow.start_run(run_name=run_name, nested=True) as _nested_run:
            # Log sampled hyperparameters
            mlflow.log_params({
                "trial_number": trial.number,
                "lr": lr,
                "batch_size": batch_size,
                "lambda_l1": lambda_l1,
                "trial_epochs": trial_epochs,
            })

            # ---- 4. Train ----
            ssim = train_one_run(cfg)

            # ---- 5. Log result ----
            mlflow.log_metric("val/mean_ssim", ssim)
            mlflow.set_tag("trial_status", "success")

        print(f"  -> Trial #{trial.number} SSIM = {ssim:.6f}")
        return ssim

    return objective


# ---------------------------------------------------------------------------
# Error-resilient wrapper -- a failed trial logs 0.0 and continues the study
# ---------------------------------------------------------------------------

class _TrialStats:
    """Simple mutable counter shared between the wrapper and main()."""
    succeeded: int = 0
    failed: int = 0

    def __init__(self):
        self.succeeded = 0
        self.failed = 0


def make_safe_objective(base_cfg: dict, trial_epochs: int, trial_dir_root: Path,
                        stats: _TrialStats):
    """Wraps the real objective so any exception becomes a failed trial (not a crash)."""
    _real = make_objective(base_cfg, trial_epochs, trial_dir_root)

    def safe_objective(trial: optuna.Trial) -> float:
        try:
            result = _real(trial)
            stats.succeeded += 1
            return result
        except Exception as exc:                          # OOM, NaN, etc.
            exc_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"\n[WARNING] Trial #{trial.number} FAILED:\n{exc_str}", file=sys.stderr)

            # Log the failure to MLflow if a run is somehow still open
            try:
                run_name = f"optuna_trial_{trial.number:03d}_FAILED"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_param("failure_reason", str(exc)[:250])
                    mlflow.set_tag("trial_status", "failed")
                    mlflow.log_metric("val/mean_ssim", 0.0)
            except Exception:
                pass  # Don't let MLflow logging errors cascade

            stats.failed += 1
            # Return a very low score so Optuna deprioritises this region but
            # the study continues (we never raise TrialPruned).
            return 0.0

    return safe_objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for the pix2pix cloud-removal model."
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_rice.yaml",
        help="Path to base training config YAML.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of Optuna trials to run (default: 20).",
    )
    parser.add_argument(
        "--trial-epochs", type=int, default=5,
        help="Epochs per trial -- keep small (3-5) for a fast sweep (default: 5).",
    )
    parser.add_argument(
        "--study-name", type=str, default="cloud-removal-hparam-sweep",
        help="Name of the Optuna study (default: cloud-removal-hparam-sweep).",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Optional SQLite/PostgreSQL URL for Optuna study persistence "
            "(e.g. sqlite:///optuna.db). Omit to use in-memory storage."
        ),
    )
    args = parser.parse_args()

    # ---- Load base config ----
    base_cfg = load_config(args.config)

    # ---- MLflow setup (mirrors train.py) ----
    mlflow_cfg = base_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    if tracking_uri and "<YOUR_" not in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get("experiment_name", "cloud-removal")
    mlflow.set_experiment(experiment_name)

    # Isolated checkpoint root so Optuna trials never clash with the baseline
    trial_dir_root = Path("models/optuna")
    trial_dir_root.mkdir(parents=True, exist_ok=True)

    # ---- Create Optuna study (maximise SSIM) ----
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    stats = _TrialStats()

    print(
        f"\nStarting Optuna sweep: {args.n_trials} trials x {args.trial_epochs} epochs each\n"
        f"Baseline (20-epoch run): SSIM = 0.8281\n"
        f"MLflow experiment: {experiment_name}\n"
        f"Tracking URI: {tracking_uri or 'local'}\n"
    )

    # ---- Run sweep inside a parent MLflow run ----
    with mlflow.start_run(run_name="optuna-sweep") as parent_run:
        mlflow.log_params({
            "n_trials": args.n_trials,
            "trial_epochs": args.trial_epochs,
            "study_name": args.study_name,
            "baseline_ssim": 0.8281,
            "lr_range": "1e-5 to 1e-3 (log)",
            "batch_size_choices": "[4, 8, 16]",
            "lambda_l1_range": "50 to 150",
        })

        safe_obj = make_safe_objective(base_cfg, args.trial_epochs, trial_dir_root, stats)
        study.optimize(safe_obj, n_trials=args.n_trials)

        # ---- Log study-level summary ----
        best_trial = study.best_trial
        best_ssim = best_trial.value
        best_params = best_trial.params

        mlflow.log_metrics({
            "sweep/best_ssim": best_ssim,
            "sweep/trials_success": stats.succeeded,
            "sweep/trials_failed": stats.failed,
        })
        mlflow.log_params({f"best/{k}": v for k, v in best_params.items()})
        mlflow.set_tag("parent_run_id", parent_run.info.run_id)

    # ---- Print summary ----
    print(
        f"\n{'='*60}\n"
        f"OPTUNA SWEEP COMPLETE\n"
        f"{'='*60}\n"
        f"  Trials run:      {args.n_trials}\n"
        f"  Succeeded:       {stats.succeeded}\n"
        f"  Failed:          {stats.failed}\n"
        f"\n  Best trial:      #{best_trial.number}\n"
        f"  Best val SSIM:   {best_ssim:.6f}  "
        f"({'BEATS' if best_ssim > 0.8281 else 'below'} baseline 0.8281)\n"
        f"\n  Best hyperparameters:\n"
        + "\n".join(f"    {k}: {v}" for k, v in best_params.items())
        + f"\n{'='*60}"
    )

    # ---- Write best config YAML (only if at least one trial succeeded) ----
    if stats.succeeded == 0:
        print("\n[WARNING] No trials succeeded -- skipping configs/train_rice_optuna_best.yaml.")
        return

    best_cfg = copy.deepcopy(base_cfg)
    best_cfg["optimizer"]["lr"] = float(best_params["lr"])
    best_cfg["training"]["batch_size"] = int(best_params["batch_size"])
    best_cfg["loss"]["lambda_l1"] = float(best_params["lambda_l1"])
    # Restore original epoch count (not the short trial value)
    # so this YAML is ready to use for a full training run.
    best_cfg["experiment"]["model_dir"] = "models/optuna_best"

    # Annotate with sweep provenance
    best_cfg.setdefault("optuna", {})
    best_cfg["optuna"]["best_trial"] = int(best_trial.number)
    best_cfg["optuna"]["best_ssim"] = float(best_ssim)
    best_cfg["optuna"]["n_trials"] = args.n_trials
    best_cfg["optuna"]["trial_epochs"] = args.trial_epochs
    best_cfg["optuna"]["baseline_ssim"] = 0.8281

    out_path = Path("configs/train_rice_optuna_best.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest config written to: {out_path}")
    print(
        f"To train with best params:\n"
        f"  python -m src.training.train --config {out_path}\n"
    )


if __name__ == "__main__":
    main()
