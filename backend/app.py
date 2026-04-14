from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import re
import time
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
CORS(app)

# Load model
pipeline = joblib.load('sentiment_model.joblib')
with open('metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

POSITIVE_WORDS_SET = set(pipeline.get('positive_words', []))
NEGATIVE_WORDS_SET = set(pipeline.get('negative_words', []))

MODEL_INFO = {
    'model_name': pipeline.get('model_name', 'sentiment_v2_tfidf_lex_lr'),
    'version': pipeline.get('version', '2.0'),
    'accuracy': pipeline['accuracy'],
    'macro_f1': pipeline['macro_f1'],
    'classes': pipeline['classes'],
    'dataset': 'synthetic_netflix_like_thai_5000',
    'language': 'Thai',
    'framework': 'TF-IDF (word+char) + Lexicon Features + Logistic Regression'
}

def preprocess_thai(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)
    return text

def lexicon_features(texts):
    features = []
    for text in texts:
        pos_count = sum(1 for w in POSITIVE_WORDS_SET if w in text)
        neg_count = sum(1 for w in NEGATIVE_WORDS_SET if w in text)
        total = pos_count + neg_count + 1e-6
        features.append([
            pos_count, neg_count,
            pos_count - neg_count,
            pos_count / total,
            neg_count / total,
            1 if pos_count > neg_count else 0,
        ])
    return np.array(features)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify(MODEL_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Empty text'}), 400

    start = time.time()
    clean = preprocess_thai(text)

    tfidf_word = pipeline['tfidf_word']
    tfidf_char = pipeline['tfidf_char']
    model = pipeline['model']

    vec_w = tfidf_word.transform([clean])
    vec_c = tfidf_char.transform([clean])
    vec_lex = csr_matrix(lexicon_features([clean]))
    vec = hstack([vec_w, vec_c, vec_lex])

    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    latency_ms = round((time.time() - start) * 1000, 2)

    classes = model.classes_.tolist()
    confidence_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    return jsonify({
        'label': pred,
        'confidence': round(float(max(proba)), 4),
        'probabilities': confidence_dict,
        'latency_ms': latency_ms,
        'model_version': MODEL_INFO['version'],
        'model_name': MODEL_INFO['model_name']
    })

@app.route('/errors', methods=['GET'])
def get_errors():
    return jsonify({
        'hard_examples': metrics.get('hard_examples', []),
        'total': len(metrics.get('hard_examples', []))
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'confusion_matrix': metrics['confusion_matrix'],
        'classes': metrics['classes'],
        'train_size': metrics.get('train_size', 4000),
        'test_size': metrics.get('test_size', 1000)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)