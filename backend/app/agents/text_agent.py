from __future__ import annotations

import logging

import scipy.sparse as sp

from app.utils.model_loader import model_loader

logger = logging.getLogger(__name__)


def vectorize_text(description: str, product_name: str = "") -> sp.csr_matrix:
    """
    Convert text description (and optional product name) into a TF-IDF sparse vector.

    The notebook trained on item_description alone. We optionally append the product
    name to enrich the text signal, matching the pattern in training data where names
    often appear alongside descriptions.
    """
    tfidf = model_loader.tfidf

    if product_name and product_name.strip():
        combined = f"{product_name.strip()}. {description or ''}"
    else:
        combined = description or ""

    if not combined.strip():
        logger.warning("Empty text input, TF-IDF will be all-zeros vector.")

    return tfidf.transform([combined])
