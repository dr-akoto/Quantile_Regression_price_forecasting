"""Day-cluster bootstrap for a development-selected calibration method.

Resample paired delivery days, not individual half-hourly observations.
This quantifies sampling variability descriptively; serial dependence
between days and a single historical test period limit interpretation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asymmetric_calibration import compare
from temporal_validation import BOUNDS, evaluate
from robustness_analysis import interval_score


def paired_day_bootstrap(predictions, development_days=60, test_days=30,
                         warmup_days=21, windows=(7, 14, 21),
                         delay_hours=48, replicates=5000, seed=2026):
    if replicates < 100:
        raise ValueError('At least 100 bootstrap replicates required')
    report = evaluate(predictions, development_days, test_days,
                      warmup_days, windows, delay_hours)
    selected = report['selected_method']
    window = report['selected_window_days']
    out, _ = compare(predictions, warmup_days=warmup_days,
                     window_days=window, delay_hours=delay_hours)
    out['day'] = pd.to_datetime(out.delivery_utc, utc=True).dt.normalize()
    split = pd.Timestamp(report['held_out_start_utc'])
    test = out.loc[out.day >= split].copy()
    if len(test) != report['held_out_selected']['observations']:
        raise ValueError('Held-out row count mismatch')
    low, high = BOUNDS[selected]
    a = test.actual.to_numpy()
    test['original_score'] = interval_score(a, test['q0.1'], test['q0.9'])
    test['selected_score'] = interval_score(a, test[low], test[high])
    test['original_covered'] = ((a >= test['q0.1']) & (a <= test['q0.9'])).astype(float)
    test['selected_covered'] = ((a >= test[low]) & (a <= test[high])).astype(float)
    test['original_width'] = test['q0.9'] - test['q0.1']
    test['selected_width'] = test[high] - test[low]
    cols = ['original_score', 'selected_score', 'original_covered',
            'selected_covered', 'original_width', 'selected_width']
    daily_sums = test.groupby('day')[cols].sum().to_numpy(dtype=float)
    daily_counts = test.groupby('day').size().to_numpy(dtype=float)
    if len(daily_counts) != test_days:
        raise ValueError('Unexpected held-out delivery days')
    # Each resampled day carries its actual number of observations.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, test_days, size=(replicates, test_days))
    totals = daily_sums[indices].sum(axis=1)
    counts = daily_counts[indices].sum(axis=1)
    means = totals / counts[:, None]
    score_delta = means[:, 1] - means[:, 0]
    coverage_delta = means[:, 3] - means[:, 2]
    width_delta = means[:, 5] - means[:, 4]

    def describe(values, point):
        lo, hi = np.quantile(values, [0.025, 0.975])
        return {'estimate': float(point), 'bootstrap_95pct_interval':
                [float(lo), float(hi)]}

    summary = {
        'selected_method': selected,
        'selected_window_days': window,
        'held_out_start_utc': report['held_out_start_utc'],
        'held_out_days': test_days,
        'held_out_observations': int(len(test)),
        'bootstrap_unit': 'UTC delivery day',
        'bootstrap_replicates': replicates,
        'random_seed': seed,
        'difference_definition': 'selected minus original',
        'interval_score_difference': describe(
            score_delta, report['held_out_selected']['mean_interval_score'] -
            report['held_out_original_reference']['mean_interval_score']),
        'coverage_difference': describe(
            coverage_delta, report['held_out_selected']['coverage'] -
            report['held_out_original_reference']['coverage']),
        'mean_width_difference': describe(
            width_delta, report['held_out_selected']['mean_width'] -
            report['held_out_original_reference']['mean_width']),
        'warning': 'Percentile intervals use iid day-cluster resampling; '
                   'serial dependence across days is not preserved. These '
                   'are exploratory uncertainty estimates, not formal '
                   'coverage guarantees or a hypothesis test. Method selection '
                   'was fixed using development dates; publication delay '
                   'remains an unverified assumption.'
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(
        'results/research_uk_daily_90d/daily_predictions.csv'))
    p.add_argument('--output', type=Path, default=Path(
        'results/research_uk_uncertainty_90d'))
    p.add_argument('--development-days', type=int, default=60)
    p.add_argument('--test-days', type=int, default=30)
    p.add_argument('--warmup-days', type=int, default=21)
    p.add_argument('--windows', type=int, nargs='+', default=[7, 14, 21])
    p.add_argument('--delay-hours', type=int, default=48)
    p.add_argument('--replicates', type=int, default=5000)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    summary = paired_day_bootstrap(
        pd.read_csv(args.input), args.development_days, args.test_days,
        args.warmup_days, args.windows, args.delay_hours,
        args.replicates, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
