"""Tests for MLflow model promotion script (scripts/promote.py) and metric gates."""

import math
import pytest
import torch.nn as nn
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from scripts.promote import evaluate_gate, promote_model, is_valid_metric, fetch_production_metrics


# --- 1. Candidate beats both production and baseline on both metrics ---
def test_gate_passes_when_strictly_beating_baseline_and_production():
    """Gate must pass when candidate strictly outperforms baseline and production on both metrics."""
    cand = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.92,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    prod = {
        "eval/mean_psnr": 26.0,
        "eval/mean_ssim": 0.88,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=prod)
    assert passes
    assert len(reasons) == 0


# --- 2. Candidate ties production PSNR ---
def test_gate_rejects_when_tying_production_psnr():
    """Gate must reject candidate if candidate PSNR ties production PSNR."""
    cand = {
        "eval/mean_psnr": 26.0,  # Exact tie with production
        "eval/mean_ssim": 0.92,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    prod = {
        "eval/mean_psnr": 26.0,
        "eval/mean_ssim": 0.88,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=prod)
    assert not passes
    assert any("not strictly greater than current Production PSNR" in r for r in reasons)


# --- 3. Candidate ties production SSIM ---
def test_gate_rejects_when_tying_production_ssim():
    """Gate must reject candidate if candidate SSIM ties production SSIM."""
    cand = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.88,  # Exact tie with production
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    prod = {
        "eval/mean_psnr": 26.0,
        "eval/mean_ssim": 0.88,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=prod)
    assert not passes
    assert any("not strictly greater than current Production SSIM" in r for r in reasons)


# --- 4. Candidate ties baseline PSNR ---
def test_gate_rejects_when_tying_baseline_psnr():
    """Gate must reject candidate if candidate PSNR ties cloudy baseline PSNR."""
    cand = {
        "eval/mean_psnr": 20.0,  # Exact tie with baseline
        "eval/mean_ssim": 0.90,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=None)
    assert not passes
    assert any("not strictly greater than baseline" in r for r in reasons)


# --- 5. Candidate ties baseline SSIM ---
def test_gate_rejects_when_tying_baseline_ssim():
    """Gate must reject candidate if candidate SSIM ties cloudy baseline SSIM."""
    cand = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.70,  # Exact tie with baseline
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=None)
    assert not passes
    assert any("not strictly greater than baseline" in r for r in reasons)


# --- 6. Candidate beats PSNR but fails SSIM ---
def test_gate_rejects_when_beating_psnr_but_failing_ssim():
    """Gate must reject candidate if it beats PSNR but underperforms on SSIM."""
    # Fails vs baseline SSIM
    cand_fail_base_ssim = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.65,  # lower than baseline 0.70
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand_fail_base_ssim, prod_metrics=None)
    assert not passes
    assert any("SSIM" in r and "baseline" in r for r in reasons)

    # Beats baseline SSIM but fails vs production SSIM
    cand_fail_prod_ssim = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.85,  # beats baseline 0.70, but lower than prod 0.88
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    prod = {
        "eval/mean_psnr": 26.0,
        "eval/mean_ssim": 0.88,
    }
    passes, reasons = evaluate_gate(cand_fail_prod_ssim, prod_metrics=prod)
    assert not passes
    assert any("SSIM" in r and "Production" in r for r in reasons)


# --- 7. Missing metrics or invalid metric values ---
def test_gate_rejects_missing_metrics():
    """Gate must reject candidate when any required metric is absent."""
    cand = {"eval/mean_psnr": 28.5}  # missing ssim and baseline
    passes, reasons = evaluate_gate(cand)
    assert not passes
    assert any("Missing required evaluation metrics" in r for r in reasons)


@pytest.mark.parametrize("invalid_val", [float("nan"), float("inf"), float("-inf"), None, "invalid"])
def test_gate_rejects_invalid_and_non_finite_metrics(invalid_val):
    """Gate must reject candidate if any metric is NaN, Inf, None, or non-numeric."""
    cand = {
        "eval/mean_psnr": invalid_val,
        "eval/mean_ssim": 0.90,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand)
    assert not passes
    assert any("invalid/non-finite" in r or "Missing" in r for r in reasons)


def test_gate_rejects_invalid_production_metrics():
    """Gate must reject promotion if current production exists but has NaN/Inf metrics."""
    cand = {
        "eval/mean_psnr": 28.0,
        "eval/mean_ssim": 0.90,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    prod = {
        "eval/mean_psnr": float("nan"),
        "eval/mean_ssim": 0.85,
    }
    passes, reasons = evaluate_gate(cand, prod_metrics=prod)
    assert not passes
    assert any("invalid/non-finite" in r for r in reasons)


# --- 8. First promotion when no production model exists ---
def test_first_promotion_when_no_production_exists():
    """When prod_metrics is None, candidate must beat baseline to pass."""
    # Beats baseline -> Passes
    cand_pass = {
        "eval/mean_psnr": 25.0,
        "eval/mean_ssim": 0.85,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand_pass, prod_metrics=None)
    assert passes
    assert len(reasons) == 0

    # Fails baseline -> Rejects
    cand_fail = {
        "eval/mean_psnr": 18.0,
        "eval/mean_ssim": 0.65,
        "eval/baseline_mean_psnr": 20.0,
        "eval/baseline_mean_ssim": 0.70,
    }
    passes, reasons = evaluate_gate(cand_fail, prod_metrics=None)
    assert not passes
    assert len(reasons) >= 1


# --- 9. End-to-end Local SQLite Promotion via Alias ---
def test_promote_end_to_end_local_sqlite(tmp_path):
    """Verify promotion workflow end-to-end on a local SQLite MLflow tracking server using aliases."""
    db_path = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Log a candidate run with required metrics and real PyTorch model artifact
    with mlflow.start_run(run_name="eval_candidate") as run:
        run_id = run.info.run_id
        mlflow.log_metrics({
            "eval/mean_psnr": 29.5,
            "eval/mean_ssim": 0.93,
            "eval/baseline_mean_psnr": 20.0,
            "eval/baseline_mean_ssim": 0.70,
        })
        dummy_model = nn.Linear(1, 1)
        mlflow.pytorch.log_model(dummy_model, artifact_path="generator", serialization_format="pickle")

    # 2. Dry run: gate passes, model not yet promoted
    promote_model(
        run_id=run_id,
        model_name="test-cloud-gan",
        alias="production",
        tracking_uri=tracking_uri,
        artifact_path="generator",
        execute_promotion=False,
    )

    # Confirm model version does not have production alias yet
    with pytest.raises(Exception):
        client.get_model_version_by_alias("test-cloud-gan", "production")

    # 3. Execute promotion via alias
    promote_model(
        run_id=run_id,
        model_name="test-cloud-gan",
        alias="production",
        tracking_uri=tracking_uri,
        artifact_path="generator",
        execute_promotion=True,
    )

    # Verify model is registered and alias 'production' points to version 1
    prod_ver = client.get_model_version_by_alias("test-cloud-gan", "production")
    assert prod_ver is not None
    assert str(prod_ver.version) == "1"
    assert "production" in prod_ver.aliases


# --- 10. Chained Promotion with Separate Training & Evaluation Runs ---
def test_promote_persists_tags_and_evaluates_second_candidate_with_separate_runs(tmp_path):
    """When training and eval runs differ, tags must persist and allow subsequent candidate comparison."""
    db_path = tmp_path / "chained_mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "chained-cloud-gan"

    # --- Candidate 1: Initial Promotion ---
    # 1a. Training run: contains model artifact, NO eval metrics
    with mlflow.start_run(run_name="cand1_train") as train_run_1:
        train_run_id_1 = train_run_1.info.run_id
        mlflow.log_metrics({"train/loss": 0.05, "val/L1_loss": 0.04})
        dummy_model_1 = nn.Linear(1, 1)
        mlflow.pytorch.log_model(dummy_model_1, artifact_path="generator", serialization_format="pickle")

    # 1b. Evaluation run: contains eval metrics, NO model artifact
    with mlflow.start_run(run_name="cand1_eval") as eval_run_1:
        eval_run_id_1 = eval_run_1.info.run_id
        mlflow.log_metrics({
            "eval/mean_psnr": 25.0,
            "eval/mean_ssim": 0.85,
            "eval/baseline_mean_psnr": 20.0,
            "eval/baseline_mean_ssim": 0.70,
        })

    # Promote Candidate 1 (first promotion, beats baseline 20.0/0.70)
    success_1 = promote_model(
        run_id=train_run_id_1,
        eval_run_id=eval_run_id_1,
        model_name=model_name,
        alias="production",
        tracking_uri=tracking_uri,
        artifact_path="generator",
        execute_promotion=True,
    )
    assert success_1 is True

    # Verify Candidate 1 is registered, has production alias, and persisted evaluation tags
    prod_ver_1 = client.get_model_version_by_alias(model_name, "production")
    assert str(prod_ver_1.version) == "1"
    assert prod_ver_1.tags.get("eval_run_id") == eval_run_id_1
    assert prod_ver_1.tags.get("eval/mean_psnr") == "25.0"
    assert prod_ver_1.tags.get("eval/mean_ssim") == "0.85"

    # --- Candidate 2: Strictly beats Candidate 1 on both metrics ---
    with mlflow.start_run(run_name="cand2_train") as train_run_2:
        train_run_id_2 = train_run_2.info.run_id
        dummy_model_2 = nn.Linear(1, 1)
        mlflow.pytorch.log_model(dummy_model_2, artifact_path="generator", serialization_format="pickle")

    with mlflow.start_run(run_name="cand2_eval") as eval_run_2:
        eval_run_id_2 = eval_run_2.info.run_id
        mlflow.log_metrics({
            "eval/mean_psnr": 27.5,   # Strictly > 25.0
            "eval/mean_ssim": 0.91,   # Strictly > 0.85
            "eval/baseline_mean_psnr": 20.0,
            "eval/baseline_mean_ssim": 0.70,
        })

    success_2 = promote_model(
        run_id=train_run_id_2,
        eval_run_id=eval_run_id_2,
        model_name=model_name,
        alias="production",
        tracking_uri=tracking_uri,
        artifact_path="generator",
        execute_promotion=True,
    )
    assert success_2 is True

    # Verify alias 'production' moved to version 2
    prod_ver_2 = client.get_model_version_by_alias(model_name, "production")
    assert str(prod_ver_2.version) == "2"
    assert prod_ver_2.tags.get("eval/mean_psnr") == "27.5"
    assert prod_ver_2.tags.get("eval/mean_ssim") == "0.91"

    # --- Candidate 3: Beats baseline, but fails against current Production (version 2) ---
    with mlflow.start_run(run_name="cand3_train") as train_run_3:
        train_run_id_3 = train_run_3.info.run_id
        dummy_model_3 = nn.Linear(1, 1)
        mlflow.pytorch.log_model(dummy_model_3, artifact_path="generator", serialization_format="pickle")

    with mlflow.start_run(run_name="cand3_eval") as eval_run_3:
        eval_run_id_3 = eval_run_3.info.run_id
        mlflow.log_metrics({
            "eval/mean_psnr": 26.0,   # Beats baseline 20.0, but loses to prod 27.5
            "eval/mean_ssim": 0.88,   # Beats baseline 0.70, but loses to prod 0.91
            "eval/baseline_mean_psnr": 20.0,
            "eval/baseline_mean_ssim": 0.70,
        })

    success_3 = promote_model(
        run_id=train_run_id_3,
        eval_run_id=eval_run_id_3,
        model_name=model_name,
        alias="production",
        tracking_uri=tracking_uri,
        artifact_path="generator",
        execute_promotion=True,
    )
    assert success_3 is False

    # Production alias MUST remain pointing to version 2
    prod_ver_current = client.get_model_version_by_alias(model_name, "production")
    assert str(prod_ver_current.version) == "2"


# --- 11. Malformed and Non-Finite Tag Rejection ---
class MockModelVersion:
    def __init__(self, version="1", run_id=None, tags=None):
        self.version = version
        self.run_id = run_id
        self.tags = tags or {}


def test_fetch_production_metrics_rejects_malformed_and_non_finite_tags():
    """fetch_production_metrics must reject missing, NaN, Inf, and malformed string tags."""
    client = MlflowClient()

    # Case 1: NaN PSNR tag
    v_nan = MockModelVersion(tags={"eval/mean_psnr": "nan", "eval/mean_ssim": "0.90"})
    m_nan = fetch_production_metrics(client, v_nan)
    assert m_nan == {}

    # Case 2: Inf SSIM tag
    v_inf = MockModelVersion(tags={"eval/mean_psnr": "28.0", "eval/mean_ssim": "inf"})
    m_inf = fetch_production_metrics(client, v_inf)
    assert m_inf == {}

    # Case 3: Malformed text tag
    v_malformed = MockModelVersion(tags={"eval/mean_psnr": "corrupt", "eval/mean_ssim": "0.90"})
    m_malformed = fetch_production_metrics(client, v_malformed)
    assert m_malformed == {}

    # Case 4: Missing SSIM tag
    v_missing = MockModelVersion(tags={"eval/mean_psnr": "28.0"})
    m_missing = fetch_production_metrics(client, v_missing)
    assert m_missing == {}

    # Case 5: Valid tags
    v_valid = MockModelVersion(tags={
        "eval/mean_psnr": "28.5",
        "eval/mean_ssim": "0.925",
        "eval/baseline_mean_psnr": "20.0",
        "eval/baseline_mean_ssim": "0.70",
    })
    m_valid = fetch_production_metrics(client, v_valid)
    assert m_valid["eval/mean_psnr"] == 28.5
    assert m_valid["eval/mean_ssim"] == 0.925
