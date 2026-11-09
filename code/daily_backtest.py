"""Daily issue-time backtest for UK day-ahead probabilistic prices.

Assumptions: UTC delivery days; forecasts issued at 12:00 UTC on D-1;
actual prices available 48 hours after delivery. Verify real publication
timing independently. Each day is refit using eligible observations only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_pipeline import QS, features, load_csv, pinball


def issue_time(day):
    return pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=12)


def eligible_training(frame, issue, delay_hours=48):
    """Keep targets known by issue; equality means publication is available."""
    return frame.loc[(frame.index + pd.Timedelta(hours=delay_hours)) <= issue]


def estimator_for(q, model, iterations):
    if model == 'linear':
        return make_pipeline(StandardScaler(), QuantileRegressor(
            quantile=q, alpha=0.1, solver='highs'))
    return HistGradientBoostingRegressor(
        loss='quantile', quantile=q, max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=35,
        l2_regularization=5, random_state=42)


def backtest(s, out, test_days=14, train_days=500, model='boosting',
             iterations=35, delay_hours=48):
    if test_days < 1 or train_days < 1 or delay_hours < 0:
        raise ValueError('test_days and train_days must be positive; delay nonnegative')
    frame = features(s)
    feature_cols = [c for c in frame.columns if c != 'target']
    # Require complete UTC delivery days; no partial-day forecasts.
    days = frame.index.normalize().unique()
    complete = [d for d in days if len(frame.loc[frame.index.normalize() == d]) == 48]
    if len(complete) < test_days:
        raise ValueError('Not enough complete delivery days for the requested backtest')
    selected = complete[-test_days:]
    predictions = []
    daily_metrics = []
    for day in selected:
        issue = issue_time(day)
        test = frame.loc[frame.index.normalize() == day].copy()
        if test[feature_cols + ['target']].isna().any().any():
            raise ValueError(f'Missing actual or forecast features for {day}')
        train = eligible_training(frame, issue, delay_hours).dropna(
            subset=feature_cols + ['target'])
        if len(train) < train_days * 48:
            raise ValueError(f'Insufficient eligible training data on {day}: '
                             f'{len(train)} rows; need {train_days * 48}')
        assert (train.index + pd.Timedelta(hours=delay_hours) <= issue).all()
        preds = []
        for q in QS:
            est = estimator_for(q, model, iterations)
            est.fit(train[feature_cols], train['target'])
            preds.append(est.predict(test[feature_cols]))
        preds = np.sort(np.vstack(preds), axis=0)
        result = pd.DataFrame({
            'delivery_utc': test.index.astype(str),
            'forecast_issue_utc': str(issue),
            'actual': test['target'].to_numpy(),
            'baseline_weekly': test['lag_336'].to_numpy(),
            'train_end_utc': str(train.index.max()),
            'train_rows': len(train),
        })
        for j, q in enumerate(QS):
            result[f'q{q:g}'] = preds[j]
        result['interval_80_covered'] = (
            (result.actual >= result['q0.1']) &
            (result.actual <= result['q0.9'])).astype(int)
        result['interval_80_width'] = result['q0.9'] - result['q0.1']
        for q in QS:
            daily_metrics.append({
                'delivery_day_utc': str(day.date()),
                'quantile': q,
                'model_pinball': pinball(result.actual, result[f'q{q:g}'], q),
                'baseline_pinball': pinball(result.actual, result.baseline_weekly, q),
                'train_rows': len(train),
                'train_end_utc': str(train.index.max()),
            })
        predictions.append(result)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(daily_metrics)
    all_predictions.to_csv(out / 'daily_predictions.csv', index=False)
    metrics.to_csv(out / 'daily_metrics.csv', index=False)
    model_loss = float(metrics.model_pinball.mean())
    baseline_loss = float(metrics.baseline_pinball.mean())
    summary = {
        'target': s.name, 'model': model, 'test_days': test_days,
        'train_days_minimum': train_days,
        'forecast_issue': '12:00 UTC on preceding UTC day',
        'assumed_publication_delay_hours': delay_hours,
        'training_policy': 'daily refit; target delivery + delay <= issue',
        'feature_columns': feature_cols,
        'mean_pinball_model': model_loss,
        'mean_pinball_weekly_baseline': baseline_loss,
        'relative_pinball_improvement_pct': float(100 * (1 - model_loss / baseline_loss))
        if baseline_loss else None,
        'empirical_80pct_coverage': float(all_predictions.interval_80_covered.mean()),
        'mean_80pct_width': float(all_predictions.interval_80_width.mean()),
        'warning': 'Assumed publication delay not independently verified. '
                   'No external point-in-time covariates. This is not HEFTCom2024 reproduction.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(__file__).parent /
                   'data' / 'Energy_Data_20200920_20231027.csv')
    p.add_argument('--time-column', default='dtm')
    p.add_argument('--target', default='DA_Price')
    p.add_argument('--timezone', default='UTC')
    p.add_argument('--output', type=Path, default=Path('results/research_uk_daily'))
    p.add_argument('--model', choices=['boosting', 'linear'], default='boosting')
    p.add_argument('--train-days', type=int, default=500)
    p.add_argument('--test-days', type=int, default=14)
    p.add_argument('--iterations', type=int, default=35)
    p.add_argument('--delay-hours', type=int, default=48)
    a = p.parse_args()
    s, meta = load_csv(a.input, a.time_column, a.target, a.timezone)
    s.name = a.target
    summary = backtest(s, a.output, a.test_days, a.train_days,
                       a.model, a.iterations, a.delay_hours)
    (a.output / 'source_audit.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
