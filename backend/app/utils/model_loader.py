from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from app.config import (
    BRAND_ENCODER_PATH,
    CAT_ENCODER_PATH,
    IMAGE_MEAN_VECTOR_PATH,
    MOBILENET_FEATURE_DIM,
    REQUIRED_MODEL_FILES,
    TFIDF_MAX_FEATURES,
    TFIDF_VECTORIZER_PATH,
    XGBOOST_MODEL_PATH,
)

if TYPE_CHECKING:
    from tensorflow.keras import Model

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class ModelLoader:
    """Singleton loader for all ML models and preprocessors."""

    _instance: Optional[ModelLoader] = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelLoader:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._tfidf: Optional[TfidfVectorizer] = None
        self._brand_encoder: Optional[LabelEncoder] = None
        self._cat_encoder: Optional[LabelEncoder] = None
        self._xgb_model: Optional[xgb.Booster] = None
        self._image_model: Optional[Model] = None
        self._global_mean_vector: Optional[np.ndarray] = None
        self._loaded = False
        self._image_model_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_image_model_loaded(self) -> bool:
        return self._image_model_loaded

    def load_text_models(self) -> None:
        """Load XGBoost, TF-IDF, and label encoders. Called at startup."""
        if self._loaded:
            return

        missing = [f for f in REQUIRED_MODEL_FILES if not f.exists()]
        if missing:
            raise ModelLoadError(
                f"Model files not found: {[str(m) for m in missing]}. "
                f"Place these files in the 'backend/models/' directory."
            )

        logger.info("Loading TF-IDF vectorizer...")
        self._tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
        if len(self._tfidf.get_feature_names_out()) != TFIDF_MAX_FEATURES:
            raise ModelLoadError(
                f"TF-IDF vocabulary size mismatch. Expected {TFIDF_MAX_FEATURES}, "
                f"got {len(self._tfidf.get_feature_names_out())}"
            )

        logger.info("Loading brand encoder...")
        self._brand_encoder = joblib.load(BRAND_ENCODER_PATH)

        logger.info("Loading category encoder...")
        self._cat_encoder = joblib.load(CAT_ENCODER_PATH)

        logger.info("Loading XGBoost model...")
        self._xgb_model = xgb.Booster()
        self._xgb_model.load_model(str(XGBOOST_MODEL_PATH))

        self._loaded = True
        logger.info("Text/metadata models loaded successfully.")

    def load_image_model(self) -> None:
        """Lazy-load MobileNetV2 on first image prediction request."""
        if self._image_model_loaded:
            return

        logger.info("Loading MobileNetV2 (first image request)...")
        import tensorflow as tf
        tf.keras.backend.clear_session()

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        self._image_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
        ])

        self._image_model_loaded = True
        logger.info("MobileNetV2 loaded successfully.")

    def load_global_mean_vector(self) -> None:
        """Load the global mean image vector for fallback."""
        if self._global_mean_vector is not None:
            return

        if IMAGE_MEAN_VECTOR_PATH.exists():
            self._global_mean_vector = np.load(IMAGE_MEAN_VECTOR_PATH)
            logger.info(f"Global mean vector loaded, shape={self._global_mean_vector.shape}")
        else:
            logger.warning(
                f"image_mean_vector.npy not found at {IMAGE_MEAN_VECTOR_PATH}. "
                f"Using zeros as fallback image features."
            )
            self._global_mean_vector = np.zeros(MOBILENET_FEATURE_DIM, dtype=np.float32)

    @property
    def tfidf(self) -> TfidfVectorizer:
        if not self._tfidf:
            raise ModelLoadError("TF-IDF not loaded. Call load_text_models() first.")
        return self._tfidf

    @property
    def brand_encoder(self) -> LabelEncoder:
        if not self._brand_encoder:
            raise ModelLoadError("Brand encoder not loaded.")
        return self._brand_encoder

    @property
    def cat_encoder(self) -> LabelEncoder:
        if not self._cat_encoder:
            raise ModelLoadError("Category encoder not loaded.")
        return self._cat_encoder

    @property
    def xgb_model(self) -> xgb.Booster:
        if not self._xgb_model:
            raise ModelLoadError("XGBoost model not loaded.")
        return self._xgb_model

    @property
    def image_model(self) -> Model:
        if not self._image_model_loaded:
            self.load_image_model()
        assert self._image_model is not None
        return self._image_model

    @property
    def global_mean_vector(self) -> np.ndarray:
        if self._global_mean_vector is None:
            self.load_global_mean_vector()
        assert self._global_mean_vector is not None
        return self._global_mean_vector


model_loader = ModelLoader()
