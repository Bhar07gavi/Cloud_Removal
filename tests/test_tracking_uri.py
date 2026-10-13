"""Tests for unified MLflow tracking URI resolution precedence."""

import os
import pytest
from serving.settings import settings, resolve_tracking_uri, load_config_tracking_uri
import scripts.promote as promote_mod
import serving.inference as inference_mod


def test_explicit_argument_takes_top_precedence(monkeypatch, tmp_path):
    """Explicit URI argument must override settings, env var, and config files."""
    dummy_yaml = tmp_path / "train.yaml"
    dummy_yaml.write_text("mlflow:\n  tracking_uri: 'https://yaml.example.com'\n", encoding="utf-8")

    monkeypatch.setattr(settings, "mlflow_tracking_uri", "sqlite:///settings.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")

    uri = resolve_tracking_uri(
        explicit_uri="sqlite:///explicit.db",
        config_path=str(dummy_yaml),
        fallback_uri="sqlite:///fallback.db",
    )
    assert uri == "sqlite:///explicit.db"


def test_settings_override_takes_precedence_over_env_and_config(monkeypatch, tmp_path):
    """Serving settings override must take precedence over generic MLFLOW_TRACKING_URI and config."""
    dummy_yaml = tmp_path / "train.yaml"
    dummy_yaml.write_text("mlflow:\n  tracking_uri: 'https://yaml.example.com'\n", encoding="utf-8")

    monkeypatch.setattr(settings, "mlflow_tracking_uri", "sqlite:///settings_override.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")

    uri = resolve_tracking_uri(
        explicit_uri=None,
        config_path=str(dummy_yaml),
        fallback_uri="sqlite:///fallback.db",
    )
    assert uri == "sqlite:///settings_override.db"


def test_mlflow_env_var_takes_precedence_over_config(monkeypatch, tmp_path):
    """MLFLOW_TRACKING_URI must take precedence over config file and fallback when settings is None."""
    dummy_yaml = tmp_path / "train.yaml"
    dummy_yaml.write_text("mlflow:\n  tracking_uri: 'https://yaml.example.com'\n", encoding="utf-8")

    monkeypatch.setattr(settings, "mlflow_tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env_custom.db")

    uri = resolve_tracking_uri(
        explicit_uri=None,
        config_path=str(dummy_yaml),
        fallback_uri="sqlite:///fallback.db",
    )
    assert uri == "sqlite:///env_custom.db"


def test_config_file_used_when_no_overrides(monkeypatch, tmp_path):
    """Config file tracking_uri must be used when no explicit arguments or env vars are set."""
    dummy_yaml = tmp_path / "train.yaml"
    dummy_yaml.write_text("mlflow:\n  tracking_uri: 'https://dagshub.example.com/mlflow'\n", encoding="utf-8")

    monkeypatch.setattr(settings, "mlflow_tracking_uri", None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    uri = resolve_tracking_uri(
        explicit_uri=None,
        config_path=str(dummy_yaml),
        fallback_uri="sqlite:///fallback.db",
    )
    assert uri == "https://dagshub.example.com/mlflow"


def test_fallback_sqlite_used_when_config_absent_or_placeholder(monkeypatch, tmp_path):
    """Fallback sqlite:///mlflow.db must be used when config file is absent or placeholder."""
    dummy_yaml = tmp_path / "placeholder.yaml"
    dummy_yaml.write_text("mlflow:\n  tracking_uri: '<YOUR_DAGSHUB_URL>'\n", encoding="utf-8")

    monkeypatch.setattr(settings, "mlflow_tracking_uri", None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Placeholder in config -> fallback
    uri1 = resolve_tracking_uri(
        explicit_uri=None,
        config_path=str(dummy_yaml),
        fallback_uri="sqlite:///default_fallback.db",
    )
    assert uri1 == "sqlite:///default_fallback.db"

    # Non-existent config -> fallback
    uri2 = resolve_tracking_uri(
        explicit_uri=None,
        config_path=str(tmp_path / "nonexistent.yaml"),
        fallback_uri="sqlite:///mlflow.db",
    )
    assert uri2 == "sqlite:///mlflow.db"


def test_promotion_and_serving_resolve_identically(monkeypatch):
    """Both scripts/promote.py and serving/inference.py must resolve to the exact same URI."""
    monkeypatch.setattr(settings, "mlflow_tracking_uri", "sqlite:///shared.db")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    uri_promote = promote_mod.resolve_tracking_uri()
    uri_serving = inference_mod.resolve_tracking_uri()
    uri_settings = resolve_tracking_uri()

    assert uri_promote == uri_serving == uri_settings == "sqlite:///shared.db"
