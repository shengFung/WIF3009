from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

from app.agents.context_agent import (
    CONDITION_MAP,
    encode_brand,
    encode_category,
    inject_condition_text,
    make_meta_features,
)
from app.agents.image_agent import extract_features
from app.agents.text_agent import vectorize_text
from app.config import RMSE_LOG_SCALE
from app.models.schemas import InputSummary, PredictionResponse
from app.utils.model_loader import model_loader

logger = logging.getLogger(__name__)


@dataclass
class PredictionInput:
    image: Optional[bytes] = None
    description: str = ""
    product_name: str = ""
    brand: str = ""
    sub_category: str = ""
    condition_id: Optional[int] = None


@dataclass
class AgentResults:
    text_vec: np.ndarray = field(default_factory=lambda: np.array([]))
    meta_vec: np.ndarray = field(default_factory=lambda: np.array([]))
    image_vec: np.ndarray = field(default_factory=lambda: np.array([]))
    image_provided: bool = False
    brand_known: bool = True
    category_known: bool = True
    warnings: list[str] = field(default_factory=list)


def run_pipeline(payload: PredictionInput) -> PredictionResponse:
    """Orchestrate the 3 agents and return a price prediction."""
    results = AgentResults()

    # --- Text Agent ---
    enriched_description = inject_condition_text(
        payload.description, payload.condition_id
    )
    text_sparse: sp.csr_matrix = vectorize_text(
        description=enriched_description,
        product_name=payload.product_name,
    )
    results.text_vec = text_sparse.toarray().astype(np.float32)

    # --- Context Agent ---
    brand_enc, results.brand_known = encode_brand(payload.brand)
    cat_enc, results.category_known = encode_category(payload.sub_category)
    results.meta_vec = make_meta_features(brand_enc, cat_enc, hype_score=0.0)

    if not results.brand_known:
        results.warnings.append(
            f"Brand '{payload.brand}' not found in training data, using fallback."
        )
    if not results.category_known:
        results.warnings.append(
            f"Category '{payload.sub_category}' not found in training data, using fallback."
        )

    # --- Image Agent ---
    image_vec, results.image_provided = extract_features(payload.image)
    results.image_vec = image_vec.reshape(1, -1).astype(np.float32)

    if not results.image_provided:
        if payload.image is None:
            results.warnings.append(
                "No image uploaded. Using average visual features — prediction based on text only."
            )
        else:
            results.warnings.append(
                "Image processing failed. Using average visual features as fallback."
            )

    # --- Feature Fusion ---
    X_final = sp.hstack([
        sp.csr_matrix(results.text_vec),
        sp.csr_matrix(results.meta_vec),
        sp.csr_matrix(results.image_vec),
    ]).tocsr()

    # --- XGBoost Predict ---
    dmatrix = xgb.DMatrix(X_final)
    log_price_pred: np.ndarray = model_loader.xgb_model.predict(dmatrix)
    predicted_price = float(np.expm1(log_price_pred[0]))

    # --- Confidence heuristic ---
    conf_margin = float(np.expm1(RMSE_LOG_SCALE))
    confidence_low = max(0.0, round(predicted_price - conf_margin, 2))
    confidence_high = round(predicted_price + conf_margin, 2)
    predicted_price = round(predicted_price, 2)

    # --- Condition label ---
    condition_label = ""
    if payload.condition_id is not None and payload.condition_id in CONDITION_MAP:
        condition_label = CONDITION_MAP[payload.condition_id]
    elif payload.condition_id is not None:
        condition_label = f"Unknown ({payload.condition_id})"

    # --- Build response ---
    input_summary = InputSummary(
        description_length=len(payload.description),
        brand=payload.brand or "",
        brand_known=results.brand_known,
        category=payload.sub_category or "",
        category_known=results.category_known,
        condition=condition_label,
        image_provided=results.image_provided,
        product_name=payload.product_name or "",
    )

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        input_summary=input_summary,
        warnings=results.warnings,
    )
