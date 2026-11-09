"""Leakage-safe rolling conformal calibration of daily backtest intervals.

Uses only prior forecast errors whose actuals would be available by issue time.
The first warmup_days are used for calibration only; all comparisons use the
same subsequent dates. Finite-sample conformal quantile is a heuristic here:
serial dependence means no distribution-free coverage guarantee is claimed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def conformal_radius(scores, alpha=0.2):
    values = np.sort(np.asarray(scores, dtype=float))
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Need finite calibration scores')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError('Insufficient calibration observations for alpha')
    return max(0.0, float(values[rank - 1]))


def calibrate(predictions, warmup_days=10, window_days=10, delay_hours=48,
              alpha=0.2):
    required = {'delivery_utc', 'forecast_issue_utc', 'actual', 'q0.1', 'q0.9'}
    if not required.issubset(predictions.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(required - set(predictions.columns))))
    if warmup_days < 1 or window_days < 1 or delay_hours < 0:
        raise ValueError('Warmup/window must be positive and delay nonnegative')
    df = predictions.copy()
    df['delivery_time'] = pd.to_datetime(df.delivery_utc, utc=True)
    df['issue_time'] = pd.to_datetime(df.forecast_issue_utc, utc=True)
    if df.delivery_time.duplicated().any():
        raise ValueError('Duplicate delivery timestamps')
    if not (df.issue_time < df.delivery_time).all():
        raise ValueError('Forecast issue must precede delivery')
    df['day'] = df.delivery_time.dt.normalize()
    df = df.sort_values('delivery_time').reset_index(drop=True)
    days = list(df.day.drop_duplicates())
    if len(days) <= warmup_days:
        raise ValueError('Need more days than warmup_days')
    output = []
    for i, day in enumerate(days):
        if i < warmup_days:
            continue
        today = df.loc[df.day == day].copy()
        if today.issue_time.nunique() != 1:
            raise ValueError('Expected one issue time per delivery day')
        issue = today.issue_time.iloc[0]
        history_days = days[max(0, i-window_days):i]
        history = df.loc[df.day.isin(history_days)].copy()
        history = history.loc[
            history.delivery_time + pd.Timedelta(hours=delay_hours) <= issue]
        if history.empty:
            raise ValueError(f'No published calibration outcomes for {day}')
        scores = np.maximum(history['q0.1'] - history.actual,
                            history.actual - history['q0.9'])
        radius = conformal_radius(scores, alpha)
        today['calibration_radius'] = radius
        today['calibration_rows'] = len(history)
        today['latest_calibration_delivery_utc'] = str(history.delivery_time.max())
        today['lower_calibrated'] = today['q0.1'] - radius
        today['upper_calibrated'] = today['q0.9'] + radius
        today['covered_original'] = (
            (today.actual >= today['q0.1']) &
            (today.actual <= today['q0.9'])).astype(int)
        today['covered_calibrated'] = (
            (today.actual >= today.lower_calibrated) &
            (today.actual <= today.upper_calibrated)).astype(int)
        output.append(today)
    result = pd.concat(output, ignore_index=True)
    original_width = result['q0.9'] - result['q0.1']
    calibrated_width = result.upper_calibrated - result.lower_calibrated
    summary = {
        'evaluation_days': len(days) - warmup_days,
        'evaluation_rows': len(result),
        'warmup_days': warmup_days,
        'rolling_window_days': window_days,
        'assumed_publication_delay_hours': delay_hours,
        'target_coverage': 1-alpha,
        'original_coverage_same_days': float(result.covered_original.mean()),
        'calibrated_coverage_same_days': float(result.covered_calibrated.mean()),
        'original_mean_width_same_days': float(original_width.mean()),
        'calibrated_mean_width_same_days': float(calibrated_width.mean()),
        'mean_calibration_radius': float(result.calibration_radius.mean()),
        'warning': 'Calibration uses assumed actuals publication delay; verify actual '
                   'publication times. Rolling dependent observations do not imply '
                   'finite-sample conformal coverage guarantees.'
    }
    return result, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path,
                   default=Path('results/research_uk_daily_30d/daily_predictions.csv'))
    p.add_argument('--output', type=Path,
                   default=Path('results/research_uk_calibration_30d'))
    p.add_argument('--warmup-days', type=int, default=10)
    p.add_argument('--window-days', type=int, default=10)
    p.add_argument('--delay-hours', type=int, default=48)
    a = p.parse_args()
    df = pd.read_csv(a.input)
    calibrated, summary = calibrate(df, a.warmup_days, a.window_days, a.delay_hours)
    a.output.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(a.output/'calibrated_predictions.csv', index=False)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
