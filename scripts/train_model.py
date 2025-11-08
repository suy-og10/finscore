"""Train a classifier for Finbyte and save a preprocessing+model pipeline.

Usage:
  python scripts/train_model.py --data data/credit_dataset.csv
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train(data_path=None, out_dir=None, random_state=42):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if data_path is None:
        data_path = os.path.join(base, 'data', 'credit_dataset.csv')
    if out_dir is None:
        out_dir = os.path.join(base, 'model')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    target = 'defaulted_12m'
    drop_cols = ['msme_id']
    X = df.drop(columns=drop_cols + [target])
    y = df[target]

    # Feature groups
    categorical = ['segment']
    numeric = [c for c in X.columns if c not in categorical]

    preproc = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('num', StandardScaler(), numeric),
        ],
        remainder='drop',
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)

    pipe = Pipeline([('preproc', preproc), ('clf', clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save pipeline
    out_file = os.path.join(out_dir, 'credit_pipeline.joblib')
    joblib.dump(pipe, out_file)
    print(f"Saved pipeline to {out_file}")

    # Feature importance: map back to feature names
    # Get feature names after OneHot encoding
    ohe = pipe.named_steps['preproc'].named_transformers_['cat']
    try:
        ohe_cols = list(ohe.get_feature_names_out(['segment']))
    except Exception:
        ohe_cols = []
    num_cols = numeric
    feature_names = list(ohe_cols) + list(num_cols)

    importances = pipe.named_steps['clf'].feature_importances_
    feature_importance = dict(zip(feature_names, [float(x) for x in importances]))

    with open(os.path.join(out_dir, 'feature_importance.json'), 'w') as f:
        json.dump({'feature_importance': feature_importance, 'accuracy': float(acc)}, f, indent=2)

    print('Wrote feature_importance.json')

    return pipe, feature_importance, acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Finbyte model')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    train(data_path=args.data, out_dir=args.out)
