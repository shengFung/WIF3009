import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Model file paths
XGBOOST_MODEL_PATH = MODEL_DIR / "hype_engine_model.json"
TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
BRAND_ENCODER_PATH = MODEL_DIR / "brand_encoder.pkl"
CAT_ENCODER_PATH = MODEL_DIR / "cat_encoder.pkl"
IMAGE_MEAN_VECTOR_PATH = MODEL_DIR / "image_mean_vector.npy"

# Image Agent settings
MOBILENET_INPUT_SIZE = (224, 224)
MOBILENET_FEATURE_DIM = 1280
TFIDF_MAX_FEATURES = 5000
META_FEATURE_COUNT = 3  # brand_enc, sub_cat1_enc, hype_score

# Server settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
RELOAD = os.environ.get("RELOAD", "true").lower() == "true"

# Prediction settings
PREDICTION_TIMEOUT_SECONDS = 30

# RMSE from training (log scale) — used for confidence range heuristic
RMSE_LOG_SCALE = 0.5442

# The notebook used these same model files
REQUIRED_MODEL_FILES = [
    XGBOOST_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    BRAND_ENCODER_PATH,
    CAT_ENCODER_PATH,
]
