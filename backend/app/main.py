from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import REQUIRED_MODEL_FILES
from app.models.schemas import HealthResponse
from app.routes.predict import router as predict_router
from app.utils.model_loader import ModelLoadError, model_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load text models. Shutdown: clean up TensorFlow session."""
    try:
        model_loader.load_text_models()
        model_loader.load_global_mean_vector()
        logger.info("All text models loaded at startup.")
    except ModelLoadError as e:
        logger.warning(f"Model loading skipped: {e}")

    yield

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        logger.info("TensorFlow session cleared.")
    except ImportError:
        pass


app = FastAPI(
    title="Apparel Price Predictor",
    description="Multi-agent AI system for resale apparel price prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check with model loading status."""
    present = [str(f.name) for f in REQUIRED_MODEL_FILES if f.exists()]
    return HealthResponse(
        status="ok" if model_loader.is_loaded else "degraded",
        models_loaded=model_loader.is_loaded,
        image_agent_loaded=model_loader.is_image_model_loaded,
        model_files_present=present,
    )


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
