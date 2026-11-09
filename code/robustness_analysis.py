"""Paired robustness diagnostics for original and calibrated 80% intervals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval_score(actual, lower, upper, alpha=0.2):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError('Non-finite values in interval inputs')
    if np.any(lo > hi):
        raise ValueError('Lower interval exceeds upper interval')
    return (hi - lo) + (2 / alpha) * np.maximum(lo - a, 0) + (2 / alpha) * np.maximum(a - hi, 0)


def diagnostics(frame, alpha=0.2):
    needed = {'delivery_utc', 'actual', 'q0.1', 'q0.9',
              'lower_calibrated', 'upper_calibrated'}
    if not needed.issubset(frame.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(needed - set(frame.columns))))
    df = frame.copy()
    df['delivery_time'] = pd.to_datetime(df.delivery_utc, utc=True)
    if df.delivery_time.duplicated().any():
        raise ValueError('Duplicate delivery timestamps')
    df['delivery_day_utc'] = df.delivery_time.dt.strftime('%Y-%m-%d')
    if df[['actual','q0.1','q0.9','lower_calibrated','upper_calibrated']].isna().any().any():
        raise ValueError('Missing actuals or bounds')
    for name, lower, upper in (
        ('original', 'q0.1', 'q0.9'),
        ('calibrated', 'lower_calibrated', 'upper_calibrated')
    ):
        a, lo, hi = df.actual, df[lower], df[upper]
        df[f'{name}_below'] = (a < lo).astype(int)
        df[f'{name}_above'] = (a > hi).astype(int)
        df[f'{name}_covered'] = ((a >= lo) & (a <= hi)).astype(int)
        df[f'{name}_width'] = hi - lo
        df[f'{name}_interval_score'] = interval_score(a, lo, hi, alpha)
    columns = [
        'original_covered', 'calibrated_covered',
        'original_below', 'calibrated_below',
        'original_above', 'calibrated_above',
        'original_width', 'calibrated_width',
        'original_interval_score', 'calibrated_interval_score'
    ]
    daily = df.groupby('delivery_day_utc', as_index=False)[columns].mean()
    daily['observations'] = df.groupby('delivery_day_utc').size().to_numpy()
    summary = {
        'evaluation_days': len(daily),
        'observations': len(df),
        'target_coverage': 1-alpha,
        'original_coverage': float(df.original_covered.mean()),
        'calibrated_coverage': float(df.calibrated_covered.mean()),
        'original_below_rate': float(df.original_below.mean()),
        'calibrated_below_rate': float(df.calibrated_below.mean()),
        'original_above_rate': float(df.original_above.mean()),
        'calibrated_above_rate': float(df.calibrated_above.mean()),
        'original_mean_width': float(df.original_width.mean()),
        'calibrated_mean_width': float(df.calibrated_width.mean()),
        'original_mean_interval_score': float(df.original_interval_score.mean()),
        'calibrated_mean_interval_score': float(df.calibrated_interval_score.mean()),
        'days_calibration_improved_coverage': int(
            (daily.calibrated_covered > daily.original_covered).sum()),
        'days_calibration_reduced_interval_score': int(
            (daily.calibrated_interval_score < daily.original_interval_score).sum()),
        'warning': 'These are paired descriptive results on the same dates, '
                   'not proof of future performance or conformal coverage guarantees.'
    }
    return daily, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(
        'results/research_uk_calibration_30d/calibrated_predictions.csv'))
    p.add_argument('--output', type=Path, default=Path(
        'results/research_uk_robustness_30d'))
    args = p.parse_args()
    daily, summary = diagnostics(pd.read_csv(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / 'daily_diagnostics.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
