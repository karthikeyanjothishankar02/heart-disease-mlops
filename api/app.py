from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
import logging
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PATH SETUP (CRITICAL FOR PICKLE)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

# ---------------------------------------------------------------------
# PROMETHEUS METRICS
# ---------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["prediction"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of active requests",
)

# ---------------------------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(MODELS_DIR, "production_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "production_preprocessor.pkl")

try:
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Model and preprocessor loaded successfully")

except Exception as exc:
    logger.exception("Failed to load model artifacts")
    raise RuntimeError("Model loading failed") from exc

# ---------------------------------------------------------------------
# FEATURE ORDER (MUST MATCH TRAINING)
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# ---------------------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------------------
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ---------------------------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------------------------
@app.middleware("http")
async def monitoring_middleware(request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Normalize endpoint (remove trailing slash)
        endpoint = request.url.path.rstrip("/")

        # Increment request counter
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()

        # Record latency
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)

        # Add process time header
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"

        return response

    finally:
        ACTIVE_REQUESTS.dec()

# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame(
            [[
                data.age, data.sex, data.cp, data.trestbps, data.chol,
                data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
                data.slope, data.ca, data.thal
            ]],
            columns=FEATURE_COLUMNS,
        )

        # Preprocess
        features_scaled = preprocessor.transform(input_df)

        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0]

        # Increment prediction counter
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        logger.info(
            "Prediction=%s Confidence=%.4f Age=%s Sex=%s",
            prediction,
            probability[prediction],
            data.age,
            data.sex,
        )

        return {
            "prediction": prediction,
            "confidence": float(probability[prediction]),
            "risk_level": "High" if prediction == 1 else "Low",
            "probabilities": {
                "no_disease": float(probability[0]),
                "disease": float(probability[1]),
            },
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
