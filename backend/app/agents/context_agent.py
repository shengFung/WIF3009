from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.utils.model_loader import model_loader

logger = logging.getLogger(__name__)

# Static condition mapping matching notebook's condition_map
CONDITION_MAP: dict[int, str] = {
    1: "New",
    2: "Used",
    3: "Used",
    4: "Worn",
    5: "Worn",
}

# Fallback label for unknown brands (matches notebook's fillna('unbranded'))
UNKNOWN_BRAND_FALLBACK = "unbranded"


def encode_brand(brand: str) -> tuple[int, bool]:
    """
    Label-encode a brand name. Returns (encoded_int, is_known).
    Falls back to 'unbranded' for unseen brands.
    """
    encoder: LabelEncoder = model_loader.brand_encoder
    le_values = list(encoder.classes_)

    if brand and brand.strip():
        brand_clean = brand.strip()
        if brand_clean in le_values:
            return int(encoder.transform([brand_clean])[0]), True
        else:
            logger.info(f"Brand '{brand_clean}' not in training data, using fallback 'unbranded'.")
            if UNKNOWN_BRAND_FALLBACK in le_values:
                return int(encoder.transform([UNKNOWN_BRAND_FALLBACK])[0]), False
            return 0, False
    else:
        if UNKNOWN_BRAND_FALLBACK in le_values:
            return int(encoder.transform([UNKNOWN_BRAND_FALLBACK])[0]), False
        return 0, False


def encode_category(sub_cat1: str) -> tuple[int, bool]:
    """
    Label-encode a sub_category. Returns (encoded_int, is_known).
    Falls back to the most common category for unseen values.
    """
    encoder: LabelEncoder = model_loader.cat_encoder
    le_values = list(encoder.classes_)

    if sub_cat1 and sub_cat1.strip():
        cat_clean = sub_cat1.strip()
        if cat_clean in le_values:
            return int(encoder.transform([cat_clean])[0]), True
        else:
            logger.info(f"Category '{cat_clean}' not in training data, using most common category.")
            return 0, False
    return 0, False


CONDITION_TEXT: dict[int, str] = {
    1: "new with tags nwt brand new never used",
    2: "like new excellent condition barely used",
    3: "good used condition minor wear",
    4: "worn condition visible wear used",
    5: "poor condition heavily worn damaged used",
}


def inject_condition_text(description: str, condition_id: int | None) -> str:
    """Append condition keywords to description so TF-IDF picks them up."""
    if condition_id is not None and condition_id in CONDITION_TEXT:
        return f"{description} condition {CONDITION_TEXT[condition_id]}"
    return description


def make_meta_features(
    brand_enc: int,
    cat_enc: int,
    hype_score: float = 0.0,
) -> np.ndarray:
    """
    Build the metadata feature array: [brand_enc, sub_cat1_enc, hype_score].

    Must return exactly the order and count the model was trained on.
    """
    return np.array([[brand_enc, cat_enc, hype_score]], dtype=np.float32)
