"""FastAPI application for satellite cloud removal serving."""

from contextlib import asynccontextmanager
import io
import logging
from typing import Optional
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch

from serving.settings import settings
from serving.inference import CloudRemovalModel, load_production_model

logger = logging.getLogger("serving.api")
logging.basicConfig(level=logging.INFO)

# Global model state
_model_instance: Optional[CloudRemovalModel] = None
_model_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load production model at startup."""
    global _model_instance, _model_load_error
    logger.info("Initializing serving application lifespan...")
    try:
        gen, version = load_production_model(
            model_name=settings.model_name,
            alias=settings.model_alias,
            tracking_uri=settings.mlflow_tracking_uri,
            device=settings.device,
            local_checkpoint_path=settings.local_checkpoint_path,
        )
        _model_instance = CloudRemovalModel(
            generator_or_path=gen,
            device=settings.device,
            version=version,
            tile_size=settings.tile_size,
            overlap=settings.tile_overlap,
        )
        _model_load_error = None
        logger.info("Successfully loaded production model: %s (version: %s)", settings.model_name, version)
    except Exception as e:
        _model_load_error = str(e)
        logger.error("Failed to load production model at startup: %s", e)
        # Note: Do not silently fallback to a dummy model. Service will start,
        # but /ready will report not ready and /predict will reject requests.

    yield

    logger.info("Shutting down serving application...")
    _model_instance = None


app = FastAPI(
    title="Satellite Cloud Removal API",
    description="Pix2Pix GAN serving endpoint for removing clouds from satellite imagery.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Liveness probe", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness probe. Always returns 200 OK while the server is running."""
    return {"status": "ok"}


@app.get("/ready", summary="Readiness probe")
def readiness_check():
    """Readiness probe. Returns 200 OK if model is loaded and ready, 503 otherwise."""
    if _model_instance is not None:
        return {
            "status": "ready",
            "model_name": settings.model_name,
            "model_version": _model_instance.version,
            "device": str(_model_instance.device),
        }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "model_name": settings.model_name,
            "error": _model_load_error or "Model is not loaded",
        },
    )


@app.post(
    "/predict",
    summary="Remove clouds from satellite image",
    responses={
        200: {"content": {"image/png": {}}, "description": "Cloud-free reconstructed image (PNG)"},
        400: {"description": "Invalid file format or corrupted image"},
        413: {"description": "Image exceeds maximum allowed resolution or file size"},
        503: {"description": "Model is not ready to serve traffic"},
    },
)
async def predict(file: UploadFile = File(..., description="Cloudy optical satellite image")):
    """Upload a cloudy satellite image and receive a reconstructed cloud-free PNG image."""
    # 1. Verify model readiness
    if _model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {_model_load_error or 'service initializing'}",
        )

    # 2. Check file size limit
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {settings.max_upload_size_bytes} bytes.",
        )

    # 3. Open and validate image format
    try:
        image = Image.open(io.BytesIO(contents))
        img_format = (image.format or "").upper()
        # Fallback to extension check if format is undetermined
        if not img_format and file.filename:
            ext = file.filename.rsplit(".", 1)[-1].upper()
            if ext in ["JPG", "JPEG"]:
                img_format = "JPEG"
            elif ext:
                img_format = ext

        if img_format not in settings.allowed_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format '{img_format}'. Allowed: {settings.allowed_formats}",
            )

        # 4. Check pixel limits
        width, height = image.size
        num_pixels = width * height
        if num_pixels > settings.max_image_pixels:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Image resolution {width}x{height} ({num_pixels:,} pixels) exceeds "
                    f"maximum allowed limit of {settings.max_image_pixels:,} pixels."
                ),
            )

        # Ensure image data is loaded into memory
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be parsed as a valid image.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {e}",
        )

    # 5. Run tiled inference
    try:
        output_image = _model_instance.predict(image)
    except Exception as e:
        logger.exception("Inference failed during prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}",
        )

    # 6. Encode output image to PNG
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Model-Name": settings.model_name,
            "X-Model-Version": str(_model_instance.version),
            "Content-Disposition": "inline; filename=cloud_free.png",
        },
    )
