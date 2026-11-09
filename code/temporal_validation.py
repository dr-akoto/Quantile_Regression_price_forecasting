"""Chronological calibration selection and held-out evaluation.

Select method/window using only development delivery dates; report final
held-out performance without reselecting the winner.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from asymmetric_calibration import compare
from robustness_analysis import interval_score

BOUNDS = {
    'original': ('q0.1', 'q0.9'),
    'symmetric': ('lower_calibrated', 'upper_calibrated'),
    'asymmetric': ('lower_asymmetric', 'upper_asymmetric')
}


def metrics(frame, method):
    lo, hi = BOUNDS[method]
    a, lower, upper = frame.actual, frame[lo], frame[hi]
    return {
        'observations': int(len(frame)),
        'coverage': float(((a >= lower) & (a <= upper)).mean()),
        'below_rate': float((a < lower).mean()),
        'above_rate': float((a > upper).mean()),
        'mean_width': float((upper - lower).mean()),
        'mean_interval_score': float(interval_score(a, lower, upper).mean())
    }


def evaluate(predictions, development_days=60, test_days=30,
             warmup_days=21, windows=(7, 14, 21), delay_hours=48):
    if min(development_days, test_days, warmup_days) < 1:
        raise ValueError('Day counts must be positive')
    if not windows or any(w < 1 for w in windows):
        raise ValueError('Calibration windows must be positive')
    df = predictions.copy()
    df['delivery_day'] = pd.to_datetime(df.delivery_utc, utc=True).dt.normalize()
    days = sorted(df.delivery_day.unique())
    if len(days) != development_days + test_days:
        raise ValueError('Expected exactly development_days + test_days distinct days')
    if development_days <= warmup_days:
        raise ValueError('Development period must exceed warmup')
    split = days[development_days]
    candidates = []
    runs = {}
    for window in sorted(set(windows)):
        out, _ = compare(predictions, warmup_days=warmup_days,
                         window_days=window, delay_hours=delay_hours)
        out['delivery_day'] = pd.to_datetime(out.delivery_utc, utc=True).dt.normalize()
        dev = out.loc[out.delivery_day < split]
        test = out.loc[out.delivery_day >= split]
        if dev.empty or test.empty:
            raise ValueError('Empty development or test split')
        runs[window] = (dev, test)
        for method in BOUNDS:
            candidates.append({
                'method': method, 'window_days': window,
                'development': metrics(dev, method)
            })
    # Original forecast is independent of window: do not duplicate it in selection.
    candidates = [c for c in candidates if c['method'] != 'original' or
                  c['window_days'] == min(windows)]
    selected = min(candidates, key=lambda c: (
        c['development']['mean_interval_score'],
        c['method'], c['window_days']))
    chosen_window = selected['window_days']
    held_out = runs[chosen_window][1]
    report = {
        'selection_metric': 'development mean interval score (lower is better)',
        'development_days': development_days,
        'test_days': test_days,
        'warmup_days': warmup_days,
        'calibration_windows': sorted(set(windows)),
        'assumed_publication_delay_hours': delay_hours,
        'development_end_utc': str(days[development_days - 1]),
        'held_out_start_utc': str(split),
        'selected_method': selected['method'],
        'selected_window_days': chosen_window,
        'development_candidates': candidates,
        'held_out_selected': metrics(held_out, selected['method']),
        'held_out_original_reference': metrics(held_out, 'original'),
        'warning': 'The selected method is fixed before held-out scoring, but '
                   'rolling calibration can use newly published outcomes during '
                   'the test period. Publication delay remains unverified; '
                   'serial dependence limits coverage guarantees.'
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(
        'results/research_uk_daily_90d/daily_predictions.csv'))
    p.add_argument('--output', type=Path, default=Path(
        'results/research_uk_temporal_validation_90d'))
    p.add_argument('--development-days', type=int, default=60)
    p.add_argument('--test-days', type=int, default=30)
    p.add_argument('--warmup-days', type=int, default=21)
    p.add_argument('--windows', type=int, nargs='+', default=[7, 14, 21])
    p.add_argument('--delay-hours', type=int, default=48)
    a = p.parse_args()
    report = evaluate(pd.read_csv(a.input), a.development_days, a.test_days,
                      a.warmup_days, a.windows, a.delay_hours)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
