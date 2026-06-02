from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from app.config import MOBILENET_FEATURE_DIM, MOBILENET_INPUT_SIZE
from app.utils.model_loader import model_loader

logger = logging.getLogger(__name__)


class ImageAgentError(Exception):
    pass


def preprocess_image(image_data: bytes, target_size: tuple[int, int] = MOBILENET_INPUT_SIZE) -> np.ndarray:
    """Load raw image bytes, preprocess into MobileNetV2-compatible array (1, 224, 224, 3)."""
    try:
        img = PILImage.open(io.BytesIO(image_data)).convert("RGB")
    except UnidentifiedImageError:
        raise ImageAgentError(
            "Image could not be processed. Please upload a valid JPEG/PNG image of the apparel item."
        )

    img = img.resize(target_size)

    try:
        import tensorflow as tf
    except ImportError:
        raise ImageAgentError("TensorFlow is not installed. Cannot process image.")

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


def extract_features(image_data: Optional[bytes]) -> tuple[np.ndarray, bool]:
    """
    Extract 1280-dim visual features from an image.

    Returns:
        (features_array, image_provided)
        - If image_data is None: returns global mean vector, image_provided=False
        - If image_data is valid: returns MobileNetV2 features, image_provided=True
        - If image processing fails: returns global mean vector, image_provided=False
    """
    if image_data is None:
        logger.info("No image provided, using global mean vector fallback.")
        return model_loader.global_mean_vector.copy(), False

    try:
        processed = preprocess_image(image_data, MOBILENET_INPUT_SIZE)
        features = model_loader.image_model.predict(processed, verbose=0)
        vector = features.flatten().astype(np.float32)

        if vector.shape[0] != MOBILENET_FEATURE_DIM:
            raise ImageAgentError(
                f"Unexpected feature dimension: {vector.shape[0]}, expected {MOBILENET_FEATURE_DIM}"
            )

        return vector, True

    except (ImageAgentError, Exception) as e:
        logger.warning(f"Image feature extraction failed: {e}. Using global mean vector fallback.")
        return model_loader.global_mean_vector.copy(), False
