"""
ablation_runner.py — Standalone ablation study script for make ablation.
"""
import sys
import os
import json

sys.path.insert(0, '.')
os.environ.setdefault('TESTING', '1')
os.environ.setdefault('CI', '1')

import joblib
import lightgbm as lgb
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from models.preprocess import generate_synthetic_baseline

MODELS_DIR = 'models'
scaler = joblib.load(f'{MODELS_DIR}/scaler.joblib')
hasher = joblib.load(f'{MODELS_DIR}/hasher.joblib')
tfidf = joblib.load(f'{MODELS_DIR}/tfidf.joblib')
meta = json.load(open(f'{MODELS_DIR}/feature_meta.json'))

df = generate_synthetic_baseline(num_samples=2000, seed=99)
le = LabelEncoder()
y = le.fit_transform(df[meta['label_col']].values)

X_num = csr_matrix(scaler.transform(df[meta['numerical_cols']].astype(float)))
X_hash = hasher.transform(
    df[meta['high_card_cols']].astype(str).to_dict(orient='records')
)
X_text = tfidf.transform(df[meta['text_col']].fillna(''))
dd = pd.get_dummies(
    df[meta['low_card_cols']], drop_first=True
).reindex(columns=meta['dummy_column_order'], fill_value=0)
X_extra = csr_matrix(
    pd.concat([dd, df[meta['bool_cols']].astype(int)], axis=1)
    .astype(float).values
)

X_full = hstack([X_num, X_hash, X_text, X_extra], format='csr')
X_no_tfidf = hstack([X_num, X_hash, X_extra], format='csr')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr, te = next(sss.split(X_full, y))

P = dict(
    objective='multiclass', n_estimators=200, num_leaves=63,
    class_weight='balanced', random_state=42, verbose=-1,
)
m1 = lgb.LGBMClassifier(**P).fit(X_full[tr], y[tr])
m2 = lgb.LGBMClassifier(**P).fit(X_no_tfidf[tr], y[tr])

f1_full = f1_score(y[te], m1.predict(X_full[te]), average='macro')
f1_no = f1_score(y[te], m2.predict(X_no_tfidf[te]), average='macro')

print(f'Full model F1      : {f1_full:.4f}')
print(f'Without TF-IDF F1  : {f1_no:.4f}')
print(f'Delta (text contrib): {f1_full - f1_no:+.4f}')

ok = f1_no >= 0.55 and f1_full <= 0.90 and (f1_full - f1_no) < 0.30
print('Ablation gates     : PASS' if ok else 'Ablation gates     : FAIL')
sys.exit(0 if ok else 1)
