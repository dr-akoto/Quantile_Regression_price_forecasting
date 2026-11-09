"""Compare rolling symmetric and asymmetric interval calibration on paired days.

Both methods use past, published forecast outcomes only. The asymmetric
method allocates 10% miscoverage to each tail, using separate conformal
corrections; it is not guaranteed to outperform symmetric calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_intervals import conformal_radius, calibrate
from robustness_analysis import interval_score


def compare(predictions, warmup_days=10, window_days=10, delay_hours=48):
    symmetric, _ = calibrate(predictions, warmup_days=warmup_days,
                              window_days=window_days, delay_hours=delay_hours)
    df = predictions.copy()
    df['delivery_time'] = pd.to_datetime(df.delivery_utc, utc=True)
    df['issue_time'] = pd.to_datetime(df.forecast_issue_utc, utc=True)
    df['day'] = df.delivery_time.dt.normalize()
    if df.delivery_time.duplicated().any():
        raise ValueError('Duplicate delivery timestamps')
    days = list(df.day.sort_values().drop_duplicates())
    corrections = {}
    for i, day in enumerate(days):
        if i < warmup_days:
            continue
        today = df.loc[df.day == day]
        if today.issue_time.nunique() != 1:
            raise ValueError('Expected one issue time per day')
        issue = today.issue_time.iloc[0]
        history_days = days[max(0, i-window_days):i]
        history = df.loc[df.day.isin(history_days)]
        history = history.loc[
            history.delivery_time + pd.Timedelta(hours=delay_hours) <= issue]
        if history.empty:
            raise ValueError('No eligible historical outcomes')
        lower_scores = history['q0.1'] - history.actual
        upper_scores = history.actual - history['q0.9']
        # Separate one-sided 10% miscoverage budgets.
        low = conformal_radius(lower_scores, alpha=0.1)
        high = conformal_radius(upper_scores, alpha=0.1)
        corrections[day] = (low, high, len(history), history.delivery_time.max())
    out = symmetric.copy()
    out['day'] = pd.to_datetime(out.delivery_utc, utc=True).dt.normalize()
    out['lower_radius_asymmetric'] = out.day.map(lambda d: corrections[d][0])
    out['upper_radius_asymmetric'] = out.day.map(lambda d: corrections[d][1])
    out['lower_asymmetric'] = out['q0.1'] - out.lower_radius_asymmetric
    out['upper_asymmetric'] = out['q0.9'] + out.upper_radius_asymmetric
    out['asymmetric_calibration_rows'] = out.day.map(lambda d: corrections[d][2])
    out['asymmetric_latest_history_utc'] = out.day.map(lambda d: str(corrections[d][3]))
    result = {'evaluation_days': len(corrections), 'observations': len(out),
              'warmup_days': warmup_days, 'window_days': window_days,
              'assumed_publication_delay_hours': delay_hours}
    for name, lo, hi in (
        ('original', 'q0.1', 'q0.9'),
        ('symmetric', 'lower_calibrated', 'upper_calibrated'),
        ('asymmetric', 'lower_asymmetric', 'upper_asymmetric')
    ):
        actual, lower, upper = out.actual, out[lo], out[hi]
        result[name] = {
            'coverage': float(((actual >= lower) & (actual <= upper)).mean()),
            'below_rate': float((actual < lower).mean()),
            'above_rate': float((actual > upper).mean()),
            'mean_width': float((upper - lower).mean()),
            'mean_interval_score': float(interval_score(actual, lower, upper).mean())
        }
    result['warning'] = ('Descriptive comparison only; rolling time dependence '
                         'and assumed publication timing preclude formal coverage guarantees.')
    return out, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(
        'results/research_uk_daily_30d/daily_predictions.csv'))
    p.add_argument('--output', type=Path, default=Path(
        'results/research_uk_asymmetric_30d'))
    p.add_argument('--warmup-days', type=int, default=10)
    p.add_argument('--window-days', type=int, default=10)
    p.add_argument('--delay-hours', type=int, default=48)
    a = p.parse_args()
    out, summary = compare(pd.read_csv(a.input), a.warmup_days,
                           a.window_days, a.delay_hours)
    a.output.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output / 'comparison_predictions.csv', index=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
