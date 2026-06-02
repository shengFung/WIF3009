from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agents.context_agent import CONDITION_MAP
from app.agents.orchestrator import PredictionInput, run_pipeline
from app.config import PREDICTION_TIMEOUT_SECONDS
from app.models.schemas import (
    BrandItem,
    CategoryItem,
    ConditionItem,
    ErrorResponse,
    PredictionResponse,
)
from app.utils.model_loader import model_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_price(
    image: Optional[UploadFile] = File(default=None),
    description: str = Form(default=""),
    product_name: str = Form(default=""),
    brand: str = Form(default=""),
    sub_category: str = Form(default=""),
    condition_id: Optional[int] = Form(default=None, ge=1, le=5),
) -> PredictionResponse:
    """
    Predict apparel resale price using the multi-agent system.

    - **image**: JPEG/PNG photo of the apparel item (optional)
    - **description**: Product description text
    - **product_name**: Product name/title (optional)
    - **brand**: Brand name (must match training vocabulary for best results)
    - **sub_category**: Sub-category (e.g., "Shoes", "Dresses")
    - **condition_id**: Item condition 1-5 (1=best, 5=worst)
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please try again.")

    image_bytes: Optional[bytes] = None
    if image is not None:
        image_bytes = await image.read()

    payload = PredictionInput(
        image=image_bytes,
        description=description,
        product_name=product_name,
        brand=brand,
        sub_category=sub_category,
        condition_id=condition_id,
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline, payload),
            timeout=PREDICTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Prediction timed out. The image processing may be slow. Try a smaller image or skip the image upload.",
        )
    except Exception as e:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return result


@router.get("/brands", response_model=list[BrandItem])
async def list_brands(q: Optional[str] = None):
    """List all known brand names. Use ?q= to filter by substring match."""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    brands = list(model_loader.brand_encoder.classes_)
    if q:
        q_lower = q.lower()
        brands = [b for b in brands if q_lower in b.lower()]
    brands_sorted = sorted(brands, key=lambda x: x.lower())
    return [BrandItem(brand=b) for b in brands_sorted]


@router.get("/categories", response_model=list[CategoryItem])
async def list_categories(q: Optional[str] = None):
    """List all known sub-categories. Use ?q= to filter by substring match."""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    categories = list(model_loader.cat_encoder.classes_)
    if q:
        q_lower = q.lower()
        categories = [c for c in categories if q_lower in c.lower()]
    categories_sorted = sorted(categories, key=lambda x: x.lower())
    return [CategoryItem(category=c) for c in categories_sorted]


@router.get("/conditions", response_model=list[ConditionItem])
async def list_conditions():
    """List condition ID to label mapping."""
    return [
        ConditionItem(condition_id=cid, label=label)
        for cid, label in CONDITION_MAP.items()
    ]
