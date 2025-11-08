"""Generate a synthetic dataset matching the Finbyte schema.

Produces a CSV at data/credit_dataset.csv under the project root.
"""
import os
import random
import math
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def make_row(i):
    segments = ['kirana', 'tailor', 'mechanic', 'pharmacy', 'salon', 'restaurant']
    msme_id = f"msme_{i:06d}"
    segment = random.choice(segments)

    # Create plausible feature values
    elec_on_time_ratio = min(max(np.random.beta(8, 2), 0.0), 1.0)
    recharge_on_time_ratio = min(max(np.random.beta(7, 3), 0.0), 1.0)
    invoice_paid_on_time_ratio = min(max(np.random.beta(6, 4), 0.0), 1.0)
    supplier_on_time_ratio = min(max(np.random.beta(7, 3), 0.0), 1.0)
    business_days_open_ratio = min(max(np.random.beta(9, 1), 0.0), 1.0)

    monthly_upi_in_count = int(np.random.poisson(30))
    monthly_upi_in_amt = float(max(1000.0, np.random.normal(50000, 25000)))
    years_in_business = max(0.1, float(np.random.exponential(5)))
    delivery_cancellations = int(np.random.poisson(1))

    avg_balance = float(max(0.0, np.random.normal(20000, 15000)))
    min_balance_freq = min(max(np.random.beta(2, 8), 0.0), 1.0)
    monthly_revenue_variance = min(max(np.random.beta(2, 5), 0.0), 1.0)

    # Logistic function to set default probability (higher risk when ratios low, cancellations high)
    score_components = (
        -2.0 * elec_on_time_ratio
        -1.8 * recharge_on_time_ratio
        -2.2 * invoice_paid_on_time_ratio
        -1.5 * supplier_on_time_ratio
        -1.0 * business_days_open_ratio
        + 0.01 * delivery_cancellations
        + 0.00001 * (50000 - avg_balance)
        + 2.0 * monthly_revenue_variance
        + 0.05 * min_balance_freq
        - 0.03 * years_in_business
    )

    p_default = sigmoid(score_components)
    defaulted_12m = int(random.random() < p_default)

    return {
        'msme_id': msme_id,
        'segment': segment,
        'elec_on_time_ratio': round(elec_on_time_ratio, 4),
        'recharge_on_time_ratio': round(recharge_on_time_ratio, 4),
        'invoice_paid_on_time_ratio': round(invoice_paid_on_time_ratio, 4),
        'supplier_on_time_ratio': round(supplier_on_time_ratio, 4),
        'business_days_open_ratio': round(business_days_open_ratio, 4),
        'monthly_upi_in_count': monthly_upi_in_count,
        'monthly_upi_in_amt': round(monthly_upi_in_amt, 2),
        'years_in_business': round(years_in_business, 2),
        'delivery_cancellations': delivery_cancellations,
        'avg_balance': round(avg_balance, 2),
        'min_balance_freq': round(min_balance_freq, 4),
        'monthly_revenue_variance': round(monthly_revenue_variance, 4),
        'defaulted_12m': defaulted_12m,
    }


def generate(n=10000, out_path=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(base, 'data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'credit_dataset.csv')

    rows = [make_row(i) for i in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic Finbyte dataset')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    generate(n=args.n, out_path=args.out)
