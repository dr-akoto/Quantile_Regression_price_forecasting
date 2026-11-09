import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from calibrate_intervals import calibrate, conformal_radius


class CalibrationTests(unittest.TestCase):
    def test_conformal_radius(self):
        self.assertEqual(conformal_radius([1, 2, 3, 4, 5, 6, 7, 8, 9], alpha=.2), 8)
        self.assertEqual(conformal_radius([-3] * 9, alpha=.2), 0)

    def test_only_published_errors_used(self):
        rows = []
        for day in pd.date_range('2023-01-01', periods=6, freq='D', tz='UTC'):
            issue = day - pd.Timedelta(hours=12)
            for hour in (0, 12):
                rows.append({
                    'delivery_utc': str(day + pd.Timedelta(hours=hour)),
                    'forecast_issue_utc': str(issue),
                    'actual': 100.0, 'q0.1': 0.0, 'q0.9': 10.0
                })
        df = pd.DataFrame(rows)
        calibrated, summary = calibrate(df, warmup_days=3, window_days=3,
                                        delay_hours=48)
        self.assertEqual(summary['evaluation_days'], 3)
        for _, row in calibrated.iterrows():
            issue = pd.Timestamp(row['issue_time'])
            last = pd.Timestamp(row['latest_calibration_delivery_utc'])
            self.assertLessEqual(last + pd.Timedelta(hours=48), issue)

    def test_no_future_outcome_leakage(self):
        rows = []
        for day in pd.date_range('2023-01-01', periods=6, freq='D', tz='UTC'):
            rows.append({
                'delivery_utc': str(day),
                'forecast_issue_utc': str(day - pd.Timedelta(hours=12)),
                'actual': 15., 'q0.1': 0., 'q0.9': 10.
            })
        df = pd.DataFrame(rows)
        base, _ = calibrate(df, warmup_days=3, window_days=3)
        altered = df.copy()
        altered.loc[altered.index >= 3, 'actual'] = 100000
        changed, _ = calibrate(altered, warmup_days=3, window_days=3)
        self.assertEqual(base.iloc[0].calibration_radius,
                         changed.iloc[0].calibration_radius)


if __name__ == '__main__':
    unittest.main()
