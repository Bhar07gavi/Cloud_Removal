"""Configuration and settings for the cloud removal serving application."""

from typing import List, Optional
from pathlib import Path
import os
import re
import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """Runtime configuration for model serving."""

    # Model Registry
    model_name: str = Field(default="cloud-removal-gan", description="Registered model name in MLflow")
    model_alias: str = Field(default="production", description="Model alias to load from registry")
    model_stage: str = Field(default="Production", description="Fallback stage for registry backends lacking alias support")
    model_uri: Optional[str] = Field(
        default=None,
        description="MLflow model URI (e.g. models:/<name>@<alias>)"
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking URI (e.g., sqlite:///mlflow.db or remote URL)"
    )
    local_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional local checkpoint override path (e.g. models/baseline/checkpoints/best.pt)"
    )

    # Compute & Inference
    device: str = Field(default="auto", description="Compute device ('cuda', 'cpu', or 'auto')")
    tile_size: int = Field(default=256, gt=0, description="Tile dimension for UNetGenerator (must be 256)")
    tile_overlap: int = Field(default=32, description="Overlap in pixels between adjacent tiles")
    tile_batch_size: int = Field(default=4, description="Batch size for parallel tile evaluation")

    # API Guardrails
    max_pixels: int = Field(
        default=25_000_000,
        gt=0,
        description="Maximum allowed image pixels (height * width) to prevent OOM/DoS"
    )
    max_upload_size_bytes: int = Field(
        default=30 * 1024 * 1024,
        description="Maximum upload size in bytes (30MB)"
    )
    allowed_formats: List[str] = Field(
        default=["PNG", "JPEG", "JPG", "TIFF", "BMP", "WEBP"],
        description="Allowed input image formats"
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("auto", "cuda", "cpu"):
            raise ValueError(f"device must be one of 'auto', 'cuda', 'cpu', got '{v}'")
        return v

    @field_validator("tile_size")
    @classmethod
    def validate_tile_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("tile_size must be > 0")
        return v

    @field_validator("max_pixels")
    @classmethod
    def validate_max_pixels(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_pixels must be > 0")
        return v

    @model_validator(mode="after")
    def validate_model_uri_and_sync(self) -> "ServingSettings":
        if self.model_uri is None:
            self.model_uri = f"models:/{self.model_name}@{self.model_alias}"
        else:
            match = re.match(r"^models:/([^/@]+)@([^/@]+)$", self.model_uri)
            if match:
                self.model_name = match.group(1)
                self.model_alias = match.group(2)
            else:
                raise ValueError("MODEL_URI must look like models:/<name>@<alias>")
        return self


settings = ServingSettings()


def load_config_tracking_uri(config_path: str = "configs/train_rice.yaml") -> Optional[str]:
    """Extract tracking URI from YAML config file if available and valid."""
    p = Path(config_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            uri = cfg.get("mlflow", {}).get("tracking_uri")
            if uri and isinstance(uri, str) and "<YOUR_" not in uri and uri.strip():
                return uri.strip()
        except Exception:
            pass
    return None


def resolve_tracking_uri(
    explicit_uri: Optional[str] = None,
    config_path: str = "configs/train_rice.yaml",
    fallback_uri: str = "sqlite:///mlflow.db",
) -> str:
    """Resolve tracking URI according to the unified project precedence policy.

    Precedence order:
    1. Explicit function/CLI argument (`explicit_uri`).
    2. Serving settings override (`settings.mlflow_tracking_uri` / `MLFLOW_TRACKING_URI`).
    3. Standard MLflow environment variable (`MLFLOW_TRACKING_URI`).
    4. Configuration file (`configs/train_rice.yaml` -> `mlflow.tracking_uri`).
    5. Fallback local backend (`fallback_uri`, default: 'sqlite:///mlflow.db').

    Returns:
        Resolved tracking URI string.
    """
    if explicit_uri and isinstance(explicit_uri, str) and explicit_uri.strip():
        return explicit_uri.strip()

    if settings.mlflow_tracking_uri and isinstance(settings.mlflow_tracking_uri, str) and settings.mlflow_tracking_uri.strip():
        return settings.mlflow_tracking_uri.strip()

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri and env_uri.strip():
        return env_uri.strip()

    cfg_uri = load_config_tracking_uri(config_path)
    if cfg_uri:
        return cfg_uri

    return fallback_uri

