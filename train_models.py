# train_models.py
# Train Thai Netflix Review Sentiment Models
# Dataset: 7.synthetic_netflix_like_thai_reviews_5000(2).csv

import re
import json
import time
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# 0. Config
# =========================

DATA_PATH = Path("data/7.synthetic_netflix_like_thai_reviews_5000.csv")

MODEL_DIR = Path("models")
METRIC_DIR = Path("metrics")

MODEL_DIR.mkdir(exist_ok=True)
METRIC_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# =========================
# 1. Preprocessing Function
# =========================

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode สำหรับภาษาไทย/สัญลักษณ์
    text = unicodedata.normalize("NFKC", text)

    # ลบ zero-width character ที่มักแอบติดมากับข้อความไทย
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Normalize ช่องว่างซ้ำให้เหลือ 1 ช่อง
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase เฉพาะกรณีมีภาษาอังกฤษปน
    text = re.sub(
        r"[A-Za-z]+",
        lambda match: match.group(0).lower(),
        text
    )

    return text


# =========================
# 2. Dataset Understanding
# =========================

print("=" * 70)
print("1. DATASET UNDERSTANDING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"จำนวนข้อมูลทั้งหมดก่อน clean: {len(df):,} แถว")
print("\nColumn ทั้งหมด:")
print(df.columns.tolist())

print("\nตัวอย่างข้อมูล 5 แถวแรกจาก dataset จริง:")
print(df[["text", "label"]].head(5))

print("\nจำนวน label แต่ละประเภท:")
print(df["label"].value_counts())

print("\nคำอธิบาย Dataset:")
print("- Dataset นี้เป็น synthetic_netflix_like_thai")
print("- ภาษา: ไทย")
print("- ประเภทข้อความ: รีวิว/ความคิดเห็นเกี่ยวกับ Netflix")
print("- Column ที่ใช้ train: text")
print("- Column label: label")
print("- งานที่ทำ: Binary Classification")
print("- Label มี 2 คลาส: Positive และ Negative")
print("- Dataset ค่อนข้าง clean และ balanced จึงไม่ทำ over-cleaning")


# =========================
# 3. Basic Data Cleaning
# =========================

print("\n" + "=" * 70)
print("2. PREPROCESSING")
print("=" * 70)

# เลือกเฉพาะ column ที่ใช้
df = df[["text", "label"]].copy()

before_dropna = len(df)
df = df.dropna(subset=["text", "label"])
after_dropna = len(df)

print(f"ลบ missing value: {before_dropna - after_dropna:,} แถว")

before_dup = len(df)
df = df.drop_duplicates(subset=["text", "label"])
after_dup = len(df)

print(f"ลบ duplicate: {before_dup - after_dup:,} แถว")

# preprocess text
df["text_clean"] = df["text"].apply(preprocess_text)

# ลบข้อความว่างหลัง preprocess
before_empty = len(df)
df = df[df["text_clean"].str.len() > 0].copy()
after_empty = len(df)

print(f"ลบข้อความว่างหลัง preprocessing: {before_empty - after_empty:,} แถว")

print(f"\nจำนวนข้อมูลหลัง clean: {len(df):,} แถว")

print("\nจำนวน label หลัง clean:")
print(df["label"].value_counts())

print("\nตัวอย่างข้อความหลัง preprocessing:")
print(df[["text", "text_clean", "label"]].head(5))


# =========================
# 4. Train/Test Split
# =========================

print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT")
print("=" * 70)

X = df["text_clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train size: {len(X_train):,} แถว")
print(f"Test size: {len(X_test):,} แถว")

print("\nTrain label distribution:")
print(y_train.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())


# =========================
# 5. Feature Extraction Config
# =========================

tfidf_config = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "max_features": 50000,
    "min_df": 2,
    "sublinear_tf": True
}


# =========================
# 6. Create 2 Model Pipelines
# =========================

print("\n" + "=" * 70)
print("4. TRAIN 2 MODELS")
print("=" * 70)

model_1 = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(**tfidf_config)
    ),
    (
        "clf",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])

model_2 = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(**tfidf_config)
    ),
    (
        "clf",
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])

models = {
    "model_1_logistic_regression": model_1,
    "model_2_linear_svm": model_2
}


# =========================
# 7. Train + Evaluate
# =========================

results = {}

for model_name, model in models.items():
    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
    report = classification_report(y_test, y_pred, digits=4)

    print(f"Train time: {train_time:.3f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nConfusion Matrix")
    print("Labels: ['Negative', 'Positive']")
    print(cm)

    print("\nClassification Report")
    print(report)

    results[model_name] = {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "train_time_seconds": round(float(train_time), 4),
        "confusion_matrix": cm.tolist(),
        "labels": ["Negative", "Positive"],
        "classification_report": report
    }

    # Save model
    model_path = MODEL_DIR / f"{model_name}.joblib"

    model_bundle = {
        "model_name": model_name,
        "model_version": f"{model_name}_v1",
        "dataset": "synthetic_netflix_like_thai",
        "language": "Thai",
        "task": "Binary Sentiment Classification",
        "labels": ["Positive", "Negative"],
        "pipeline": model,
        "preprocessing": {
            "missing_value": "drop rows with missing text/label",
            "duplicate": "drop duplicated text/label rows",
            "unicode_normalization": "NFKC",
            "whitespace_normalization": True,
            "lowercase": "only English characters",
            "over_cleaning": False,
            "emoji_slang_removed": False
        },
        "feature_extraction": {
            "method": "TF-IDF",
            "level": "word-level",
            "ngram_range": [1, 2],
            "max_features": 50000,
            "min_df": 2,
            "sublinear_tf": True
        },
        "metrics": results[model_name]
    }

    joblib.dump(model_bundle, model_path)
    print(f"\nSaved model to: {model_path}")


# =========================
# 8. Compare Models
# =========================

print("\n" + "=" * 70)
print("5. MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame([
    {
        "model": name,
        "accuracy": info["accuracy"],
        "macro_f1": info["macro_f1"],
        "train_time_seconds": info["train_time_seconds"]
    }
    for name, info in results.items()
])

comparison_df = comparison_df.sort_values(
    by=["macro_f1", "accuracy"],
    ascending=False
)

print(comparison_df)

best_model_name = comparison_df.iloc[0]["model"]

print(f"\nBest model selected for deployment: {best_model_name}")


# =========================
# 9. Save Metrics
# =========================

comparison_path = METRIC_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

results_path = METRIC_DIR / "training_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset": "7.synthetic_netflix_like_thai_reviews_5000(2).csv",
            "total_rows_after_cleaning": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "best_model": best_model_name,
            "models": results
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nSaved comparison to: {comparison_path}")
print(f"Saved training results to: {results_path}")


# =========================
# 10. Save Error Examples
# =========================

print("\n" + "=" * 70)
print("6. ERROR ANALYSIS EXAMPLES")
print("=" * 70)

best_model = models[best_model_name]
best_pred = best_model.predict(X_test)

error_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "predicted_label": best_pred
})

error_df = error_df[error_df["true_label"] != error_df["predicted_label"]]

error_path = METRIC_DIR / "error_examples.csv"
error_df.head(20).to_csv(error_path, index=False, encoding="utf-8-sig")

print(f"จำนวนตัวอย่างที่ทำนายผิด: {len(error_df):,} แถว")
print(f"Saved error examples to: {error_path}")

if len(error_df) > 0:
    print("\nตัวอย่างที่โมเดลทำนายผิด:")
    print(error_df.head(10))
else:
    print("\nไม่พบตัวอย่างที่ทำนายผิดใน test set")
    print("หมายเหตุ: Dataset อาจเป็น synthetic และมี pattern ของ Positive/Negative ที่แยกกันค่อนข้างชัด")


print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)