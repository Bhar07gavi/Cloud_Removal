"""MLflow Model Registry Promotion CLI with strict metric gates.

Usage:
    python scripts/promote.py --run-id <RUN_ID> [--promote] [--tracking-uri <URI>]

Gates:
    1. Candidate must have valid, finite PSNR and SSIM test evaluation metrics.
    2. Candidate must strictly outperform the cloudy-input baseline on BOTH PSNR and SSIM.
    3. Candidate must strictly outperform the current production model on BOTH PSNR and SSIM (if one exists).
    4. Promotion is dry-run by default; requires explicit '--promote' flag.
"""

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Ensure repository root is on sys.path for standalone script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from serving.settings import resolve_tracking_uri, load_config_tracking_uri


def fetch_run_metrics(client: MlflowClient, run_id: str) -> Dict[str, float]:
    """Retrieve evaluation metrics for a specific MLflow run."""
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        raise RuntimeError(f"Could not fetch MLflow run '{run_id}': {e}") from e


def is_valid_metric(val: Any) -> bool:
    """Check if value is a valid, non-null, finite float."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def evaluate_gate(
    cand_metrics: Dict[str, float],
    prod_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[bool, List[str]]:
    """Strictly evaluate candidate against baseline and current production metrics.

    Requirements:
    - Candidate metrics must be present and finite.
    - Candidate PSNR > Baseline PSNR (strict).
    - Candidate SSIM > Baseline SSIM (strict).
    - If a production model exists, its metrics must be present and finite.
    - Candidate PSNR > Production PSNR (strict).
    - Candidate SSIM > Production SSIM (strict).
    - Any tie (equality) fails the gate.
    """
    reasons = []

    # 1. Candidate and Baseline metric presence & validity check
    required_metrics = [
        "eval/mean_psnr",
        "eval/mean_ssim",
        "eval/baseline_mean_psnr",
        "eval/baseline_mean_ssim",
    ]
    missing = [m for m in required_metrics if m not in cand_metrics]
    if missing:
        return False, [f"Missing required evaluation metrics in candidate run: {missing}"]

    for m in required_metrics:
        if not is_valid_metric(cand_metrics.get(m)):
            return False, [f"Candidate metric '{m}' has invalid/non-finite value: {cand_metrics.get(m)}"]

    cand_psnr = float(cand_metrics["eval/mean_psnr"])
    cand_ssim = float(cand_metrics["eval/mean_ssim"])
    base_psnr = float(cand_metrics["eval/baseline_mean_psnr"])
    base_ssim = float(cand_metrics["eval/baseline_mean_ssim"])

    # 2. Strict baseline comparison (equality fails)
    if cand_psnr <= base_psnr:
        reasons.append(
            f"Candidate PSNR ({cand_psnr:.4f}) is not strictly greater than baseline ({base_psnr:.4f})"
        )
    if cand_ssim <= base_ssim:
        reasons.append(
            f"Candidate SSIM ({cand_ssim:.6f}) is not strictly greater than baseline ({base_ssim:.6f})"
        )

    # 3. Existing production model comparison (if one exists)
    if prod_metrics is not None:
        prod_required = ["eval/mean_psnr", "eval/mean_ssim"]
        prod_missing = [m for m in prod_required if m not in prod_metrics]
        if prod_missing:
            return False, [
                f"Current production model has missing evaluation metrics: {prod_missing}. "
                "Cannot reliably verify improvement."
            ]

        for m in prod_required:
            if not is_valid_metric(prod_metrics.get(m)):
                return False, [
                    f"Current production metric '{m}' has invalid/non-finite value: {prod_metrics.get(m)}. "
                    "Cannot reliably verify improvement."
                ]

        prod_psnr = float(prod_metrics["eval/mean_psnr"])
        prod_ssim = float(prod_metrics["eval/mean_ssim"])

        # Strict production comparison (equality fails)
        if cand_psnr <= prod_psnr:
            reasons.append(
                f"Candidate PSNR ({cand_psnr:.4f}) is not strictly greater than current Production PSNR ({prod_psnr:.4f})"
            )
        if cand_ssim <= prod_ssim:
            reasons.append(
                f"Candidate SSIM ({cand_ssim:.6f}) is not strictly greater than current Production SSIM ({prod_ssim:.6f})"
            )

    passes = len(reasons) == 0
    return passes, reasons


def print_comparison_table(
    cand_metrics: Dict[str, float],
    prod_metrics: Optional[Dict[str, float]] = None,
    candidate_id: str = "Candidate",
):
    """Print formatted markdown comparison table of candidate vs baseline vs production."""
    base_psnr = cand_metrics.get("eval/baseline_mean_psnr", float("nan"))
    base_ssim = cand_metrics.get("eval/baseline_mean_ssim", float("nan"))
    cand_psnr = cand_metrics.get("eval/mean_psnr", float("nan"))
    cand_ssim = cand_metrics.get("eval/mean_ssim", float("nan"))

    print("\n" + "=" * 65)
    print(f"{'Model / Checkpoint':<25} | {'Mean PSNR (dB)':<16} | {'Mean SSIM':<16}")
    print("-" * 65)
    print(f"{'Cloudy Baseline':<25} | {base_psnr:<16.4f} | {base_ssim:<16.6f}")

    if prod_metrics and "eval/mean_psnr" in prod_metrics:
        p_psnr = prod_metrics["eval/mean_psnr"]
        p_ssim = prod_metrics["eval/mean_ssim"]
        print(f"{'Current Production':<25} | {p_psnr:<16.4f} | {p_ssim:<16.6f}")

    delta_psnr = cand_psnr - base_psnr
    delta_ssim = cand_ssim - base_ssim
    print(f"{candidate_id[:25]:<25} | {cand_psnr:<16.4f} | {cand_ssim:<16.6f}")
    print(f"{'  Delta vs Baseline':<25} | {delta_psnr:+16.4f} | {delta_ssim:+16.6f}")
    print("=" * 65 + "\n")


def fetch_production_metrics(client: MlflowClient, prod_ver: Any) -> Dict[str, float]:
    """Retrieve evaluation metrics for the production model version.

    Preference order:
    1. Registered training run (prod_ver.run_id) if it contains valid, finite evaluation metrics.
    2. Stored validated evaluation metrics in model version tags ('eval/mean_psnr', 'eval/mean_ssim').
    3. Tagged evaluation run ID ('eval_run_id') if present.

    Returns:
        Dict of metric name -> float. Malformed or non-finite values are filtered out.
    """
    # 1. Inspect registered training run
    if prod_ver.run_id:
        try:
            run_metrics = fetch_run_metrics(client, prod_ver.run_id)
            if (
                "eval/mean_psnr" in run_metrics
                and "eval/mean_ssim" in run_metrics
                and is_valid_metric(run_metrics.get("eval/mean_psnr"))
                and is_valid_metric(run_metrics.get("eval/mean_ssim"))
            ):
                print(f"Using evaluation metrics from registered production run '{prod_ver.run_id}'.")
                return run_metrics
            else:
                print(
                    f"Notice: Registered run '{prod_ver.run_id}' does not contain required evaluation metrics. "
                    "Checking stored model version tags."
                )
        except Exception as e:
            print(f"Notice: Could not fetch metrics directly from registered run '{prod_ver.run_id}': {e}")

    # 2. Prefer stored validated evaluation metrics from model version tags
    tags = getattr(prod_ver, "tags", {}) or {}
    tag_psnr = tags.get("eval/mean_psnr")
    tag_ssim = tags.get("eval/mean_ssim")

    if is_valid_metric(tag_psnr) and is_valid_metric(tag_ssim):
        print(f"Using stored validated evaluation metrics from production model version tags (version {prod_ver.version}).")
        stored_metrics = {
            "eval/mean_psnr": float(tag_psnr),
            "eval/mean_ssim": float(tag_ssim),
        }
        if is_valid_metric(tags.get("eval/baseline_mean_psnr")):
            stored_metrics["eval/baseline_mean_psnr"] = float(tags["eval/baseline_mean_psnr"])
        if is_valid_metric(tags.get("eval/baseline_mean_ssim")):
            stored_metrics["eval/baseline_mean_ssim"] = float(tags["eval/baseline_mean_ssim"])
        return stored_metrics
    elif tag_psnr is not None or tag_ssim is not None:
        print(
            f"Warning: Model version tags contain malformed or non-finite metrics: "
            f"eval/mean_psnr={tag_psnr}, eval/mean_ssim={tag_ssim}. Ignoring invalid tags."
        )

    # 3. Fallback to tagged evaluation run ID if present
    eval_run_id = tags.get("eval_run_id")
    if eval_run_id:
        try:
            eval_metrics = fetch_run_metrics(client, eval_run_id)
            if (
                "eval/mean_psnr" in eval_metrics
                and "eval/mean_ssim" in eval_metrics
                and is_valid_metric(eval_metrics.get("eval/mean_psnr"))
                and is_valid_metric(eval_metrics.get("eval/mean_ssim"))
            ):
                print(f"Using evaluation metrics from tagged evaluation run '{eval_run_id}'.")
                return eval_metrics
        except Exception as e:
            print(f"Notice: Could not fetch metrics from tagged evaluation run '{eval_run_id}': {e}")

    print(f"Warning: No valid evaluation metrics found for production version {prod_ver.version}.")
    return {}


def promote_model(
    run_id: str,
    eval_run_id: Optional[str] = None,
    model_name: str = "cloud-removal-gan",
    alias: str = "production",
    tracking_uri: Optional[str] = None,
    artifact_path: str = "generator",
    execute_promotion: bool = False,
    exit_on_failure: bool = False,
) -> bool:
    """Evaluate and optionally promote model to the registry via the production alias."""
    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    client = MlflowClient(tracking_uri=resolved_uri)

    print(f"Tracking URI: {resolved_uri}")
    print(f"Inspecting candidate model run: {run_id}")

    metric_source_run_id = eval_run_id or run_id
    if eval_run_id:
        print(f"Retrieving candidate evaluation metrics from run: {metric_source_run_id}")

    # 1. Fetch candidate metrics from the specified run
    cand_metrics = fetch_run_metrics(client, metric_source_run_id)

    # 2. Check whether a production model currently exists via alias
    prod_metrics = None
    has_production_model = False
    prod_ver = None
    try:
        prod_ver = client.get_model_version_by_alias(model_name, alias)
        has_production_model = True
    except Exception:
        has_production_model = False

    if has_production_model and prod_ver is not None:
        print(f"Found current production model version {prod_ver.version} (run: {prod_ver.run_id})")
        prod_metrics = fetch_production_metrics(client, prod_ver)
    else:
        print(f"No existing model found with alias '{alias}'. This will be a first promotion.")

    # 3. Display comparison
    print_comparison_table(cand_metrics, prod_metrics, candidate_id=f"Run {metric_source_run_id}")

    # 4. Evaluate promotion gate strictly
    passes, failure_reasons = evaluate_gate(cand_metrics, prod_metrics)

    if not passes:
        print("[REJECTED] Candidate failed promotion gate:")
        for r in failure_reasons:
            print(f"  - {r}")
        if exit_on_failure:
            sys.exit(1)
        return False

    print("[PASSED] Candidate strictly outperforms all gate criteria!")

    if not execute_promotion:
        print("\nDry-run complete. Model was NOT promoted.")
        print(f"To promote this candidate to '{alias}', re-run with '--promote':")
        print(f"    python scripts/promote.py --run-id {run_id} --promote")
        return True

    # 5. Execute Promotion via MLflow Model Alias
    print(f"\nRegistering model '{model_name}' from run '{run_id}'...")
    model_source = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_source, model_name)
    version_num = model_version.version
    print(f"Registered model version {version_num} for '{model_name}'.")

    # Set alias 'production' as the single source of truth
    client.set_registered_model_alias(model_name, alias, str(version_num))
    print(f"Set alias '{alias}' -> version {version_num}.")

    # Persist evaluation run ID and validated candidate PSNR/SSIM as model version tags
    client.set_model_version_tag(model_name, str(version_num), "eval_run_id", metric_source_run_id)
    client.set_model_version_tag(model_name, str(version_num), "eval/mean_psnr", str(cand_metrics["eval/mean_psnr"]))
    client.set_model_version_tag(model_name, str(version_num), "eval/mean_ssim", str(cand_metrics["eval/mean_ssim"]))
    if "eval/baseline_mean_psnr" in cand_metrics:
        client.set_model_version_tag(model_name, str(version_num), "eval/baseline_mean_psnr", str(cand_metrics["eval/baseline_mean_psnr"]))
    if "eval/baseline_mean_ssim" in cand_metrics:
        client.set_model_version_tag(model_name, str(version_num), "eval/baseline_mean_ssim", str(cand_metrics["eval/baseline_mean_ssim"]))
    print(f"Persisted validated evaluation tags on model version {version_num}.")
    print(f"\nSUCCESS: Candidate {run_id} successfully promoted to '{model_name}@{alias}'.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict MLflow Model Registry Promotion CLI")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID containing the model artifact")
    parser.add_argument("--eval-run-id", type=str, default=None, help="MLflow run ID containing evaluation metrics (defaults to --run-id)")
    parser.add_argument("--model-name", type=str, default="cloud-removal-gan", help="Registered model name")
    parser.add_argument("--alias", type=str, default="production", help="Target alias (default: 'production')")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--artifact-path", type=str, default="generator", help="Logged artifact subpath")
    parser.add_argument("--promote", action="store_true", help="Perform actual promotion (manual confirmation)")
    args = parser.parse_args()

    success = promote_model(
        run_id=args.run_id,
        eval_run_id=args.eval_run_id,
        model_name=args.model_name,
        alias=args.alias,
        tracking_uri=args.tracking_uri,
        artifact_path=args.artifact_path,
        execute_promotion=args.promote,
        exit_on_failure=True,
    )
    if not success:
        sys.exit(1)

