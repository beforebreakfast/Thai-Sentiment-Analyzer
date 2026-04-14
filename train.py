"""
Thai Sentiment Model v2
Dataset: 7.synthetic_netflix_like_thai_reviews_5000.csv
รันคำสั่ง: python train.py
Output: backend/sentiment_model.joblib, backend/metrics.json
"""

import pandas as pd
import numpy as np
import re
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix

# ============================================================
# Thai Sentiment Lexicon
# ============================================================
POSITIVE_WORDS = [
    'ดี', 'ดีมาก', 'เยี่ยม', 'ยอดเยี่ยม', 'สุดยอด', 'ชอบ', 'ชอบมาก',
    'ดีงาม', 'น่าประทับใจ', 'ประทับใจ', 'สนุก', 'สนุกมาก', 'น่าสนใจ',
    'แนะนำ', 'แนะนำให้ดู', 'คุ้มค่า', 'คุ้มเวลา', 'ถูกใจ', 'โดนใจ',
    'น่าดู', 'ติดตาม', 'ติดใจ', 'อิ่มเอม', 'ดีเกินคาด', 'เกินคาด',
    'ลงตัว', 'ทำถึง', 'น่าชม', 'ดีเลิศ', 'เพอร์เฟกต์', 'perfect',
    'เจ๋ง', 'โคตรดี', 'ปัง', 'ปังมาก', 'ว้าว', 'wow', 'amazing',
    'excellent', 'great', 'good', 'awesome', 'fantastic', 'wonderful',
    'อยากให้มีซีซั่นต่อ', 'ดูจบแล้วอิ่มเอม', 'แนะนำให้ดูเลย', 'ติดเลย',
    'คุ้มเวลามาก', 'masterpiece', 'รักเลย',
]

NEGATIVE_WORDS = [
    'แย่', 'แย่มาก', 'ห่วย', 'ห่วยมาก', 'ไม่ดี', 'ไม่ชอบ', 'ผิดหวัง',
    'น่าเบื่อ', 'เบื่อ', 'เสียเวลา', 'เสียตังค์', 'ไม่แนะนำ', 'อย่าดู',
    'ไม่คุ้ม', 'ไม่โอเค', 'พัง', 'ขยะ', 'trash', 'bad', 'terrible',
    'awful', 'boring', 'worst', 'horrible', 'disappointing', 'waste',
    'ไม่ไหว', 'น่าผิดหวัง', 'ไม่น่าสนใจ', 'negative', 'Negative',
    'ผิดหวัง', 'ไม่ค่อยไหว', 'คงไม่ดูซ้ำ', 'หวังว่าจะทำดีกว่านี้',
    'ไม่แนะนำเท่าไร', 'จบแล้วเฉยๆ', 'หลวมๆ',
]

POSITIVE_WORDS_SET = set(POSITIVE_WORDS)
NEGATIVE_WORDS_SET = set(NEGATIVE_WORDS)

# ============================================================
# Augmented Training Data (ประโยคทั่วไป นอกเหนือจาก dataset)
# ============================================================
AUGMENT_DATA = [
    ('ดีมาก แนะนำเลย', 'Positive'),
    ('แย่มาก ไม่แนะนำ', 'Negative'),
    ('negative มากๆ เลย', 'Negative'),
    ('positive มากๆ', 'Positive'),
    ('ห่วยมาก ไม่คุ้มเลย', 'Negative'),
    ('ยอดเยี่ยมมาก ชอบมากเลย', 'Positive'),
    ('เสียเวลามาก ไม่ดีเลย', 'Negative'),
    ('สนุกมาก ดูจบแล้วยังอยากดูต่อ', 'Positive'),
    ('เบื่อมาก ไม่น่าสนใจเลย', 'Negative'),
    ('ปังมาก โคตรดี', 'Positive'),
    ('แย่สุดๆ เลย worst เลย', 'Negative'),
    ('ดีงาม ถูกใจมาก', 'Positive'),
    ('ไม่โอเค เสียตังค์เปล่า', 'Negative'),
    ('น่าประทับใจมาก ชอบมากเลย', 'Positive'),
    ('น่าผิดหวังมาก ไม่คุ้มค่าเลย', 'Negative'),
    ('bad movie ไม่ชอบเลย', 'Negative'),
    ('great movie ชอบมากเลย', 'Positive'),
    ('terrible ห่วยมาก', 'Negative'),
    ('amazing wonderful ดีมากเลย', 'Positive'),
    ('boring ไม่น่าสนใจเลย', 'Negative'),
    ('excellent film ดีมาก', 'Positive'),
    ('disappointing ผิดหวังมาก', 'Negative'),
    ('awesome สนุกมากๆ', 'Positive'),
    ('waste of time เสียเวลา', 'Negative'),
    ('ไม่ดีเลย แย่มากๆ', 'Negative'),
    ('ดีมากเลย ชอบมาก', 'Positive'),
    ('ห่วยมากๆ ไม่แนะนำ', 'Negative'),
    ('สุดยอดมาก แนะนำเลย', 'Positive'),
    ('ขยะ trash ไม่ดีเลย', 'Negative'),
    ('ยอดเยี่ยม เกินคาดมาก', 'Positive'),
] * 10  # repeat 10x เพื่อให้ weight สูงขึ้น


def preprocess_thai(text):
    """Preprocessing ขั้นต่ำ: normalize whitespace + ลด repeated punctuation"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)
    return text


def lexicon_features(texts):
    """สร้าง feature จากการนับคำ positive/negative ใน lexicon"""
    features = []
    for text in texts:
        pos = sum(1 for w in POSITIVE_WORDS_SET if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS_SET if w in text)
        total = pos + neg + 1e-6
        features.append([
            pos,
            neg,
            pos - neg,          # net sentiment score
            pos / total,        # positive ratio
            neg / total,        # negative ratio
            1 if pos > neg else 0,  # majority vote
        ])
    return np.array(features)


# ============================================================
# 1. โหลด Dataset
# ============================================================
print("📂 โหลด dataset...")
df = pd.read_csv('7.synthetic_netflix_like_thai_reviews_5000.csv')
print(f"   รวม {len(df)} แถว | labels: {df['label'].value_counts().to_dict()}")

df['text_clean'] = df['text'].apply(preprocess_thai)

# ============================================================
# 2. แบ่ง Train / Test
# ============================================================
X_main = df['text_clean'].tolist()
y_main = df['label'].tolist()

X_train_main, X_test, y_train_main, y_test = train_test_split(
    X_main, y_main, test_size=0.2, random_state=42, stratify=y_main
)

# เพิ่ม augmented data เข้า train เท่านั้น (ไม่ปนกับ test)
aug_df = pd.DataFrame(AUGMENT_DATA, columns=['text_clean', 'label'])
X_train = X_train_main + aug_df['text_clean'].tolist()
y_train = y_train_main + aug_df['label'].tolist()

print(f"\n📊 Train: {len(X_train)} (dataset: {len(X_train_main)}, augmented: {len(aug_df)})")
print(f"   Test:  {len(X_test)} (dataset เท่านั้น)")

# ============================================================
# 3. Vectorize
# ============================================================
print("\n🔢 สร้าง TF-IDF features...")
tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3),
                              max_features=40000, sublinear_tf=True, min_df=1)
tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                              max_features=30000, sublinear_tf=True, min_df=1)

X_train_vec = hstack([
    tfidf_word.fit_transform(X_train),
    tfidf_char.fit_transform(X_train),
    csr_matrix(lexicon_features(X_train)),
])
X_test_vec = hstack([
    tfidf_word.transform(X_test),
    tfidf_char.transform(X_test),
    csr_matrix(lexicon_features(X_test)),
])

# ============================================================
# 4. Train Model
# ============================================================
print("🚀 เทรนโมเดล Logistic Regression...")
model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, random_state=42)
model.fit(X_train_vec, y_train)

# ============================================================
# 5. Evaluate
# ============================================================
y_pred = model.predict(X_test_vec)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average='macro')
cm   = confusion_matrix(y_test, y_pred)

print(f"\n✅ ผลการประเมิน:")
print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"   Macro F1 : {f1:.4f}")
print(f"   Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred)}")

# Hard examples (low confidence)
y_proba = model.predict_proba(X_test_vec)
conf    = np.max(y_proba, axis=1)
wrong   = np.where(np.array(y_pred) != np.array(y_test))[0]
low_c   = np.argsort(conf)[:20]
hard    = []
seen    = set()
for i in list(wrong[:10]) + list(low_c[:10]):
    if i in seen: continue
    seen.add(i)
    hard.append({
        'text': X_test[i], 'true': y_test[i], 'pred': y_pred[i],
        'confidence': float(conf[i]), 'correct': bool(y_pred[i] == y_test[i]),
        'error_type': 'mixed signal' if 'แต่' in X_test[i] or 'ทว่า' in X_test[i]
                      else 'negation' if 'ไม่' in X_test[i] else 'ambiguous'
    })

# ============================================================
# 6. บันทึกโมเดลและ metrics
# ============================================================
print("💾 บันทึกไฟล์...")
pipeline = {
    'tfidf_word': tfidf_word,
    'tfidf_char': tfidf_char,
    'model': model,
    'classes': list(model.classes_),
    'version': '2.0',
    'accuracy': float(acc),
    'macro_f1': float(f1),
    'model_name': 'sentiment_v2_tfidf_lex_lr',
    'lexicon_features': True,
    'positive_words': list(POSITIVE_WORDS_SET),
    'negative_words': list(NEGATIVE_WORDS_SET),
}
joblib.dump(pipeline, 'backend/sentiment_model.joblib')
print("   ✓ backend/sentiment_model.joblib")

metrics = {
    'accuracy': float(acc), 'macro_f1': float(f1),
    'confusion_matrix': cm.tolist(), 'classes': list(model.classes_),
    'hard_examples': hard, 'train_size': len(X_train_main), 'test_size': len(X_test)
}
with open('backend/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("   ✓ backend/metrics.json")
print("\n🎉 เสร็จแล้ว! รัน backend ด้วย: cd backend && python app.py")