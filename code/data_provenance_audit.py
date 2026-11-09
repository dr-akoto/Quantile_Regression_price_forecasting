"""Audit UK forecasting input without inventing its market provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(path, time_column='dtm', target='DA_Price', timezone='UTC',
          assumed_delay_hours=48, declared_unit=None, source_url=None):
    if assumed_delay_hours < 0:
        raise ValueError('Publication delay must be nonnegative')
    path = Path(path)
    raw = pd.read_csv(path)
    if time_column not in raw or target not in raw:
        raise ValueError('Missing requested timestamp or target column')
    parsed = pd.to_datetime(raw[time_column], errors='coerce', utc=False)
    if parsed.isna().any():
        raise ValueError('Invalid timestamps: ' + str(int(parsed.isna().sum())))
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        # Explicitly reject DST ambiguous/nonexistent naive local timestamps.
        parsed = parsed.dt.tz_localize(timezone, ambiguous='raise',
                                      nonexistent='raise')
    utc = parsed.dt.tz_convert('UTC')
    values = pd.to_numeric(raw[target], errors='coerce')
    ordered = pd.DataFrame({'delivery': utc, 'price': values}).sort_values('delivery')
    duplicate_count = int(ordered.delivery.duplicated().sum())
    start, end = ordered.delivery.min(), ordered.delivery.max()
    grid = pd.date_range(start, end, freq='30min', tz='UTC')
    missing = grid.difference(pd.DatetimeIndex(ordered.delivery))
    off_grid = int((~ordered.delivery.isin(grid)).sum())
    uk_days = ordered.delivery.dt.tz_convert('Europe/London').dt.date
    day_counts = ordered.groupby(uk_days).size()
    # Only interior local dates are expected to have complete settlement days.
    interior = day_counts.iloc[1:-1] if len(day_counts) > 2 else day_counts.iloc[0:0]
    unexpected = {}
    for date, count in interior.items():
        local_start = pd.Timestamp(date).tz_localize('Europe/London')
        local_end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize('Europe/London')
        expected = int((local_end.tz_convert('UTC') - local_start.tz_convert('UTC')) /
                       pd.Timedelta(minutes=30))
        if count != expected:
            unexpected[str(date)] = {'observed': int(count), 'expected': expected}
    numeric = values.dropna()
    return {
        'input_path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'timestamp_column': time_column,
        'target_column': target,
        'declared_timezone_for_naive_timestamps': timezone,
        'rows': int(len(raw)),
        'columns': list(raw.columns),
        'first_delivery_utc': str(start),
        'last_delivery_utc': str(end),
        'duplicate_utc_timestamps': duplicate_count,
        'missing_30min_utc_grid_timestamps': int(len(missing)),
        'first_missing_utc_timestamps': [str(x) for x in missing[:10]],
        'off_30min_grid_timestamps': off_grid,
        'missing_or_non_numeric_target_values': int(values.isna().sum()),
        'negative_price_count': int((numeric < 0).sum()),
        'zero_price_count': int((numeric == 0).sum()),
        'price_min': float(numeric.min()) if len(numeric) else None,
        'price_max': float(numeric.max()) if len(numeric) else None,
        'uk_local_days_with_unexpected_counts_excluding_endpoints': unexpected,
        'declared_price_unit': declared_unit,
        'source_url_supplied_by_user': source_url,
        'source_provenance_verified': False,
        'publication_delay_hours_assumed': assumed_delay_hours,
        'publication_timing_verified': False,
        'open_questions': [
            'Identify the original data publisher and download URL',
            'Verify DA_Price market definition, currency and energy unit',
            'Verify delivery timestamp semantics and original timezone',
            'Verify historical price publication and revision timestamps',
            'Verify data licensing and redistribution restrictions'
        ],
        'warning': 'Checks data structure only. A user-supplied source URL or '
                   'unit is a declaration, not independent verification.'
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(__file__).parent /
                   'data' / 'Energy_Data_20200920_20231027.csv')
    p.add_argument('--time-column', default='dtm')
    p.add_argument('--target', default='DA_Price')
    p.add_argument('--timezone', default='UTC')
    p.add_argument('--delay-hours', type=int, default=48)
    p.add_argument('--declared-unit')
    p.add_argument('--source-url')
    p.add_argument('--output', type=Path, default=Path('results/research_uk_data_audit'))
    args = p.parse_args()
    report = audit(args.input, args.time_column, args.target, args.timezone,
                   args.delay_hours, args.declared_unit, args.source_url)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'data_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
