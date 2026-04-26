from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.preprocess import build_input_text, detect_noise_and_difficulty

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METRIC_DIR = ROOT / "metrics"
STATIC_DIR = ROOT / "static"

DATA_CANDIDATES = [
    ROOT / "data" / "7.synthetic_netflix_like_thai_reviews_5000.csv",
]

MODEL_FILES = {
    "model_1_logistic_regression": MODEL_DIR / "model_1_logistic_regression.joblib",
    "model_2_linear_svm": MODEL_DIR / "model_2_linear_svm.joblib",
}

DEFAULT_BEST_MODEL_KEY = "model_1_logistic_regression"

app = FastAPI(
    title="Thai Sentiment Analyzer — Netflix Reviews",
    version="1.2.1",
    description="FastAPI backend for Thai sentiment prediction with A/B model comparison.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODELS: dict[str, Any] = {}
COMPARISON: dict[str, Any] = {}
SAMPLE_DF: pd.DataFrame | None = None


class PredictRequest(BaseModel):
    body: str = Field(default="", description="Main Thai review text")
    compare: bool = Field(default=True, description="Return both model outputs for A/B comparison")
    model_key: str = Field(
        default="auto",
        description="auto, model_1_logistic_regression, or model_2_linear_svm",
    )


class PredictResult(BaseModel):
    model_key: str
    model_display_name: str
    model_version: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


def get_pipeline(bundle: Any) -> Any:
    """Support both saved formats: a bundle dict with ['pipeline'] or a raw sklearn Pipeline."""
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle["pipeline"]
    return bundle


def get_bundle_value(bundle: Any, key: str, default: Any = None) -> Any:
    if isinstance(bundle, dict):
        return bundle.get(key, default)
    return default


def get_best_model_key() -> str:
    """Return a valid best model key even if old metrics JSON still contains old model names."""
    best_key = COMPARISON.get("best_model_key") or COMPARISON.get("best_model") or DEFAULT_BEST_MODEL_KEY
    if best_key in MODELS:
        return best_key

    # Backward-compatible mapping from old app versions.
    old_to_new = {
        "model_a_word_tfidf_lr": "model_1_logistic_regression",
        "model_b_char_tfidf_svm": "model_2_linear_svm",
        "model_b_char_tfidf_lr": "model_2_linear_svm",
        "model_logistic_regression": "model_1_logistic_regression",
        "model_linear_svc": "model_2_linear_svm",
    }
    mapped = old_to_new.get(str(best_key))
    if mapped in MODELS:
        return mapped

    return DEFAULT_BEST_MODEL_KEY


def get_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise HTTPException(
        status_code=404,
        detail="ไม่พบไฟล์ dataset สำหรับสุ่มตัวอย่าง กรุณาตรวจสอบไฟล์ CSV ในโฟลเดอร์ data/",
    )


def load_models() -> None:
    global MODELS, COMPARISON

    missing = [str(path) for path in MODEL_FILES.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Model files not found. Run `py train_models.py` before starting the app. Missing: "
            + ", ".join(missing)
        )

    MODELS = {key: joblib.load(path) for key, path in MODEL_FILES.items()}

    metrics_json = METRIC_DIR / "model_comparison.json"
    training_json = METRIC_DIR / "training_results.json"

    if metrics_json.exists():
        COMPARISON = json.loads(metrics_json.read_text(encoding="utf-8"))
    elif training_json.exists():
        COMPARISON = json.loads(training_json.read_text(encoding="utf-8"))
    else:
        COMPARISON = {"best_model_key": DEFAULT_BEST_MODEL_KEY, "models": []}


def load_sample_dataset() -> pd.DataFrame:
    global SAMPLE_DF

    if SAMPLE_DF is None:
        data_path = get_data_path()
        df = pd.read_csv(data_path)

        required_cols = {"text", "label"}
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=500, detail="ไฟล์ dataset ไม่มีคอลัมน์ text/label ที่จำเป็น")

        SAMPLE_DF = df.dropna(subset=["text", "label"]).copy()

    return SAMPLE_DF


@app.on_event("startup")
def startup_event() -> None:
    load_models()
    load_sample_dataset()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "thai-sentiment-analyzer",
        "models_loaded": sorted(MODELS.keys()),
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    available_models = []

    for key, bundle in MODELS.items():
        pipeline = get_pipeline(bundle)
        classes = list(getattr(pipeline, "classes_", []))

        available_models.append(
            {
                "model_key": key,
                "model_display_name": get_bundle_value(bundle, "model_display_name", key),
                "model_version": get_bundle_value(bundle, "model_version", key),
                "classes": get_bundle_value(bundle, "classes", classes),
                "preprocessing": get_bundle_value(bundle, "preprocessing", {}),
                "metrics": {
                    "accuracy": get_bundle_value(bundle, "metrics", {}).get("accuracy")
                    if isinstance(get_bundle_value(bundle, "metrics", {}), dict)
                    else None,
                    "macro_f1": get_bundle_value(bundle, "metrics", {}).get("macro_f1")
                    if isinstance(get_bundle_value(bundle, "metrics", {}), dict)
                    else None,
                },
            }
        )

    return {
        "app": "Thai Sentiment Analyzer — Netflix Reviews",
        "best_model_key": get_best_model_key(),
        "available_models": available_models,
        "api": ["GET /health", "GET /model/info", "POST /predict", "GET /examples", "GET /errors"],
    }


@app.get("/examples")
def examples(
    label: str = Query(default="any", description="Positive, Negative, or any"),
    n: int = Query(default=1, ge=1, le=20),
) -> dict[str, Any]:
    df = load_sample_dataset()
    label_value = label.strip()

    if label_value.lower() != "any":
        matched = df[df["label"].astype(str).str.lower() == label_value.lower()]
        if matched.empty:
            raise HTTPException(status_code=404, detail=f"ไม่พบตัวอย่าง label: {label}")
    else:
        matched = df

    samples = matched.sample(n=min(n, len(matched))).to_dict(orient="records")

    items = [
        {
            "review_id": item.get("review_id"),
            "text": item.get("text"),
            "label": item.get("label"),
            "source": item.get("source"),
            "language": item.get("language"),
        }
        for item in samples
    ]

    return {"items": items, "source": "training dataset", "requested_label": label_value}


def sigmoid(x: float) -> float:
    """Safe sigmoid for LinearSVC decision_function confidence approximation."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def estimate_probabilities(pipeline: Any, text: str, predicted_label: str) -> dict[str, float]:
    classes = list(pipeline.classes_)

    # Logistic Regression has predict_proba().
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        return {label: round(float(prob), 4) for label, prob in zip(classes, probs)}

    # LinearSVC has decision_function(), not predict_proba().
    # For binary classification, decision_function > 0 usually means classes_[1].
    if hasattr(pipeline, "decision_function"):
        decision = pipeline.decision_function([text])[0]

        if len(classes) == 2:
            p_class_1 = sigmoid(float(decision))
            probs = {
                classes[0]: 1 - p_class_1,
                classes[1]: p_class_1,
            }
            return {label: round(float(prob), 4) for label, prob in probs.items()}

    # Fallback when probability cannot be computed.
    return {
        label: 1.0 if label == predicted_label else 0.0
        for label in classes
    }


def predict_one(model_key: str, text: str) -> PredictResult:
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")

    bundle = MODELS[model_key]
    pipeline = get_pipeline(bundle)

    started = time.perf_counter()

    predicted_label = pipeline.predict([text])[0]
    probabilities = estimate_probabilities(pipeline, text, predicted_label)

    confidence = probabilities.get(predicted_label, 0.0)
    latency_ms = (time.perf_counter() - started) * 1000

    return PredictResult(
        model_key=model_key,
        model_display_name=get_bundle_value(bundle, "model_display_name", model_key),
        model_version=get_bundle_value(bundle, "model_version", model_key),
        label=predicted_label,
        confidence=round(float(confidence), 4),
        probabilities=probabilities,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    text = build_input_text("", payload.body)

    if not text:
        raise HTTPException(status_code=400, detail="กรุณากรอกข้อความรีวิว")

    request_start = time.perf_counter()

    if payload.compare:
        selected_keys = ["model_1_logistic_regression", "model_2_linear_svm"]
    elif payload.model_key == "auto":
        selected_keys = [get_best_model_key()]
    else:
        selected_keys = [payload.model_key]

    results = [predict_one(key, text).dict() for key in selected_keys]

    best_key = get_best_model_key()
    primary = next((result for result in results if result["model_key"] == best_key), results[0])

    total_latency_ms = (time.perf_counter() - request_start) * 1000

    return {
        "ok": True,
        "input": {
            "body": payload.body,
            "combined_text": text,
            "detected_tags": detect_noise_and_difficulty(text),
        },
        "primary_result": primary,
        "results": results,
        "model_version": primary["model_version"],
        "latency_ms": round(total_latency_ms, 3),
    }


@app.get("/errors")
def errors(limit: int = 10) -> dict[str, Any]:
    path = METRIC_DIR / "error_examples.csv"

    if not path.exists():
        return {
            "items": [],
            "mode": "missing",
            "message": "ยังไม่มีไฟล์ error_examples.csv กรุณา train model ก่อน",
        }

    df = pd.read_csv(path)

    if "is_true_error" in df.columns:
        true_df = df[df["is_true_error"].astype(str).str.lower().isin(["true", "1"])]

        if not true_df.empty:
            shown = true_df.head(max(1, min(limit, 50)))
            return {
                "items": shown.to_dict(orient="records"),
                "mode": "true_errors",
                "message": "แสดงเฉพาะตัวอย่างที่โมเดลทำนายผิดจริงจากชุด test",
            }

    shown = df.head(max(1, min(limit, 50)))

    return {
        "items": shown.to_dict(orient="records"),
        "mode": "low_confidence_cases",
        "message": "ชุด test รอบนี้ไม่พบ error จริง จึงแสดงเคสที่โมเดลมี confidence ต่ำที่สุดเพื่อใช้วิเคราะห์ความเสี่ยงแทน",
    }

# py -m uvicorn app:app --reload