# Thai Sentiment Analyzer — Netflix Reviews

## Overview
This project is a **Thai Sentiment Analyzer** for Netflix-like reviews. It uses **TF-IDF** and **Logistic Regression** with **Linear SVM** for sentiment classification. The system predicts whether a review is **Positive** or **Negative** and compares both models' performance through A/B testing.

## Features
- **Review Input:** Users can input Thai text reviews.
- **Random Sample:** Get random review samples from the training dataset.
- **Prediction:** Predicts sentiment (Positive/Negative).
- **Model Comparison:** A/B comparison between **Logistic Regression** and **Linear SVM** models.
- **Error Analysis:** View low-confidence cases.

## Dataset
- **File:** `7.synthetic_netflix_like_thai_reviews_5000.csv`
- **Classes:** Positive, Negative
- **Total Rows:** 5000 (2500 Positive, 2500 Negative)
- **Tags:** `clean/noisy`, `easy/hard`

## Preprocessing
- Normalizes text (Unicode, whitespace).
- Converts English characters to lowercase.
- Does not remove emojis, slang, or negation (important for sentiment).

## Models
1. **Logistic Regression** with TF-IDF (Word-level)
2. **Linear SVM** with TF-IDF (Word-level)

## Training
- Split data 80% train, 20% test.
- Both models achieved **100% accuracy** on the test set.
- Model 1 (Logistic Regression) is selected for deployment.

## Web Application
- Built with **FastAPI**.
- APIs: `/health`, `/model/info`, `/predict`, `/examples`, `/errors`
- Frontend: HTML, CSS, JS to interact with models and display predictions.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
