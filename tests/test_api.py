"""Tests for FastAPI serving endpoints (/health, /ready, /predict)."""

import io
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from starlette.testclient import TestClient

import serving.api as api_module
from serving.api import app
from serving.inference import CloudRemovalModel
from serving.settings import settings


class DummyDeterministicGenerator(nn.Module):
    """Simple deterministic generator for fast, isolated API testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def client():
    """Create Starlette/FastAPI TestClient with lifespan disabled for explicit state tests."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def inject_test_model(monkeypatch):
    """Inject a deterministic model directly into API state without external dependencies."""
    dummy_gen = DummyDeterministicGenerator()
    test_model = CloudRemovalModel(
        generator_or_path=dummy_gen,
        device="cpu",
        version="v1.0.0-test",
        tile_size=256,
        overlap=32,
    )
    monkeypatch.setattr(api_module, "_model_instance", test_model)
    monkeypatch.setattr(api_module, "_model_load_error", None)
    return test_model


@pytest.fixture
def unload_model(monkeypatch):
    """Ensure API model is unloaded to test not-ready states."""
    monkeypatch.setattr(api_module, "_model_instance", None)
    monkeypatch.setattr(api_module, "_model_load_error", "No production model found in registry")


def make_test_image_bytes(width: int = 256, height: int = 256, fmt: str = "PNG") -> bytes:
    """Generate in-memory valid image bytes."""
    arr = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# --- Liveness and Readiness Tests ---

def test_health_endpoint(client):
    """GET /health must always return 200 OK while server is alive."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_endpoint_when_not_loaded(client, unload_model):
    """GET /ready must return 503 when model is not loaded."""
    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert "error" in data


def test_ready_endpoint_when_loaded(client, inject_test_model):
    """GET /ready must return 200 with model metadata when model is ready."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_version"] == "v1.0.0-test"
    assert data["model_name"] == settings.model_name
    assert "device" in data


# --- Prediction Error Handling Tests ---

def test_predict_model_not_loaded(client, unload_model):
    """POST /predict must reject requests with 503 if model is not loaded."""
    img_bytes = make_test_image_bytes(256, 256)
    resp = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_predict_empty_file(client, inject_test_model):
    """POST /predict must return 400 on empty file payload."""
    resp = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_predict_invalid_corrupted_file(client, inject_test_model):
    """POST /predict must return 400 when file is not a valid image."""
    resp = client.post(
        "/predict",
        files={"file": ("garbage.png", b"this is not an image", "image/png")},
    )
    assert resp.status_code == 400
    assert "valid image" in resp.json()["detail"].lower()


def test_predict_oversized_image_pixels(client, inject_test_model, monkeypatch):
    """POST /predict must return 413 when image pixels exceed max_pixels limit."""
    # Set limit artificially low for test
    monkeypatch.setattr(settings, "max_pixels", 10_000)

    # 200x200 = 40,000 pixels > 10,000 limit
    img_bytes = make_test_image_bytes(200, 200)
    resp = client.post(
        "/predict",
        files={"file": ("oversized.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


# --- Prediction Success Tests ---

def test_predict_success_png_response(client, inject_test_model):
    """POST /predict must return 200, PNG media type, X-Model-Version, and valid PNG bytes."""
    width, height = 300, 400
    img_bytes = make_test_image_bytes(width, height, fmt="PNG")

    resp = client.post(
        "/predict",
        files={"file": ("cloudy.png", img_bytes, "image/png")},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["x-model-version"] == "v1.0.0-test"
    assert resp.headers["x-model-name"] == settings.model_name

    # Validate output image dimensions match original input
    out_img = Image.open(io.BytesIO(resp.content))
    assert out_img.format == "PNG"
    assert out_img.size == (width, height)


def test_predict_jpeg_input_returns_png(client, inject_test_model):
    """POST /predict must accept JPEG and always return PNG."""
    img_bytes = make_test_image_bytes(256, 256, fmt="JPEG")

    resp = client.post(
        "/predict",
        files={"file": ("satellite.jpg", img_bytes, "image/jpeg")},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    out_img = Image.open(io.BytesIO(resp.content))
    assert out_img.format == "PNG"
    assert out_img.size == (256, 256)
