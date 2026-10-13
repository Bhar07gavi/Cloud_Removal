"""Inference pipeline for cloud removal serving with MLflow registry integration."""

from typing import Optional, Tuple, Union
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image

from src.utils.device import get_device
from src.models.generator import UNetGenerator
from src.inference.tiling import TiledInferenceEngine
from serving.settings import settings, resolve_tracking_uri


def load_production_model(
    model_name: Optional[str] = None,
    alias: Optional[str] = None,
    stage: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    device: Optional[str] = None,
    local_checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """Load the production model strictly from MLflow Registry or a specified local checkpoint.

    Returns:
        Tuple of (nn.Module generator, version_string)

    Raises:
        RuntimeError: if the production model cannot be resolved or loaded.
    """
    target_device = get_device(device or settings.device)
    target_model_name = model_name or settings.model_name
    target_alias = alias or settings.model_alias
    target_stage = stage or settings.model_stage
    resolved_uri = resolve_tracking_uri(tracking_uri)

    # 1. Check local checkpoint override first if explicitly provided
    chk_path = local_checkpoint_path or settings.local_checkpoint_path
    if chk_path and Path(chk_path).exists():
        print(f"Loading generator from local checkpoint: {chk_path} on {target_device}")
        generator = UNetGenerator(in_channels=3, out_channels=3)
        checkpoint = torch.load(str(chk_path), map_location=target_device)
        state_dict = checkpoint.get("generator", checkpoint)
        generator.load_state_dict(state_dict)
        generator.to(target_device)
        generator.eval()
        return generator, f"local:{Path(chk_path).name}"

    # 2. MLflow Registry resolution
    try:
        import mlflow
        import mlflow.pytorch
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise RuntimeError(
            f"MLflow is not installed; cannot load from registry '{target_model_name}'. "
            "Install mlflow or specify a local_checkpoint_path."
        ) from e

    if resolved_uri:
        mlflow.set_tracking_uri(resolved_uri)

    client = MlflowClient(tracking_uri=resolved_uri)
    model_version_str = "unknown"
    model_uri = None

    # MLflow Model Alias lookup (e.g. 'production')
    try:
        model_version = client.get_model_version_by_alias(target_model_name, target_alias)
        model_version_str = str(model_version.version)
        model_uri = f"models:/{target_model_name}@{target_alias}"
    except Exception as alias_err:
        raise RuntimeError(
            f"No production model found with alias '{target_alias}' for registered model '{target_model_name}' "
            f"in MLflow registry ({resolved_uri or 'default'}): {alias_err}"
        ) from alias_err

    print(f"Loading production model from MLflow URI '{model_uri}' (version {model_version_str})...")
    generator = mlflow.pytorch.load_model(model_uri, map_location=target_device)
    generator.to(target_device)
    generator.eval()
    return generator, model_version_str


class CloudRemovalModel:
    """Production Cloud Removal Model wrapper with tiled inference."""

    def __init__(
        self,
        generator_or_path: Optional[Union[nn.Module, str]] = None,
        device: str = "auto",
        version: str = "1",
        tile_size: int = 256,
        overlap: int = 32,
    ):
        self.device = get_device(device)
        self.version = version

        if isinstance(generator_or_path, nn.Module):
            self.generator = generator_or_path.to(self.device)
            self.generator.eval()
        elif isinstance(generator_or_path, str) and Path(generator_or_path).exists():
            print(f"Loading model from file: {generator_or_path}...")
            self.generator = UNetGenerator(in_channels=3, out_channels=3)
            checkpoint = torch.load(generator_or_path, map_location=self.device)
            state_dict = checkpoint.get("generator", checkpoint)
            self.generator.load_state_dict(state_dict)
            self.generator.to(self.device)
            self.generator.eval()
            self.version = Path(generator_or_path).name
        elif generator_or_path is None:
            # Load production model via registry / settings
            gen, ver = load_production_model(device=device)
            self.generator = gen
            self.version = ver
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {generator_or_path}")

        self.tiling_engine = TiledInferenceEngine(
            generator=self.generator,
            device=self.device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=settings.tile_batch_size,
        )

    def predict(
        self,
        img: Image.Image,
        tile_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Image.Image:
        """Run tiled inference, preserving exact image dimensions."""
        if tile_size is not None or overlap is not None:
            engine = TiledInferenceEngine(
                generator=self.generator,
                device=self.device,
                tile_size=tile_size or self.tiling_engine.tile_size,
                overlap=overlap if overlap is not None else self.tiling_engine.overlap,
                batch_size=self.tiling_engine.batch_size,
            )
            return engine.predict_pil(img)
        return self.tiling_engine.predict_pil(img)
