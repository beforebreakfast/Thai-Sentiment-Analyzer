# Thai Sentiment Analyzer — Netflix Reviews

MLDS Assignment: Machine Learning → Evaluation → Deployment  
Dataset: `7.synthetic_netflix_like_thai_reviews_5000`

---

## Project Structure

```
DS-Project/
├── backend/
│   ├── app.py                  # Flask API
│   ├── sentiment_model.joblib  # Trained model
│   ├── metrics.json            # Evaluation metrics
│   └── requirements.txt
└── frontend/
    └── index.html              # Web UI (single file)
└── 7.synthetic_netflix_like_thai_reviews_5000.csv
└── train.py
```

---

## Setup & Run

### Backend (API)

```bash
cd webapp/backend
pip install -r requirements.txt
python app.py
```

API runs at: `http://localhost:5000`

### Frontend

Open `webapp/frontend/index.html` in a browser.

> **Note**: The frontend works in demo/offline mode if the backend is not running. For full functionality (real predictions), start the Flask backend first.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| GET | `/metrics` | Evaluation metrics |
| GET | `/errors` | Hard/error examples |
| POST | `/predict` | Sentiment prediction |

### POST /predict

**Request:**
```json
{"text": "ซีรีส์เรื่องนี้ดีมาก ดูจบแล้วอิ่มเอม"}
```

**Response:**
```json
{
  "label": "Positive",
  "confidence": 0.9876,
  "probabilities": {"Positive": 0.9876, "Negative": 0.0124},
  "latency_ms": 3.5,
  "model_version": "1.0",
  "model_name": "sentiment_v1_tfidf_lr"
}
```

---

## Model Details

- **Vectorizer**: TF-IDF (word 1-2gram + char 2-4gram combined)
- **Classifier**: Logistic Regression (C=1.0, class_weight=balanced)
- **Train/Test Split**: 80/20 (4000 / 1000)
- **Accuracy**: 100% (synthetic dataset with clear patterns)
- **Macro F1**: 100%

## Dataset

- Language: Thai
- Type: Netflix-style movie/series reviews
- Labels: Positive / Negative (binary, balanced)
- Size: 5,000 samples

## Preprocessing Steps

1. **Whitespace normalization**: collapse multiple spaces/newlines → single space
2. **Repeated punctuation reduction**: `!!!!!!` → `!!` (preserve expression, reduce noise)

Over-cleaning (removing emojis, slang) was intentionally avoided per assignment guidelines.
