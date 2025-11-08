"""Model helper: load pipeline and expose a predict_score function.

The pipeline file expected: ../model/credit_pipeline.joblib
"""
import os
from typing import Dict, Any, List
import numpy as np
import joblib
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'credit_pipeline.joblib')


def load_pipeline(path: str = None):
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def score_from_proba(p_default: float) -> tuple[int, str]:
    """Compute score and band from default probability."""
    score = int(round((1.0 - float(p_default)) * 100))
    if score >= 80:
        band = 'Low'
    elif score >= 60:
        band = 'Medium'
    else:
        band = 'High'
    return score, band


def predict_one(pipeline, features: Dict[str, Any]):
    # Ensure column ordering consistent with training; pipeline expects the same columns used during training
    input_order = ['segment', 'elec_on_time_ratio', 'recharge_on_time_ratio', 'invoice_paid_on_time_ratio',
                   'supplier_on_time_ratio', 'business_days_open_ratio', 'monthly_upi_in_count',
                   'monthly_upi_in_amt', 'years_in_business', 'delivery_cancellations', 'avg_balance',
                   'min_balance_freq', 'monthly_revenue_variance']

    row = {c: [features.get(c)] for c in input_order}
    df = pd.DataFrame(row)
    probs = pipeline.predict_proba(df)[0]
    # probs: [p_not_default, p_default]
    p_default = float(probs[1])
    score, band = score_from_proba(p_default)
    # Map band to legacy quality label for app compatibility
    quality = 'Good' if band == 'Low' else ('Average' if band == 'Medium' else 'Poor')
    return {
        'p_default': p_default,
        'score': score,
        'band': band,
        'quality': quality,
    }
