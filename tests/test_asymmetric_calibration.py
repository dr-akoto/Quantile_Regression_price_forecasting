import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from asymmetric_calibration import compare


class AsymmetricCalibrationTests(unittest.TestCase):
    @staticmethod
    def sample():
        rows = []
        for day in pd.date_range('2023-01-01', periods=7, freq='D', tz='UTC'):
            for halfhour in range(48):
                rows.append({
                    'delivery_utc': str(day + pd.Timedelta(minutes=30 * halfhour)),
                    'forecast_issue_utc': str(day - pd.Timedelta(hours=12)),
                    'actual': float(halfhour),
                    'q0.1': 5., 'q0.9': 35.
                })
        return pd.DataFrame(rows)

    def test_three_methods_same_observations(self):
        out, summary = compare(self.sample(), warmup_days=4, window_days=4)
        self.assertEqual(summary['evaluation_days'], 3)
        self.assertEqual(summary['observations'], 144)
        self.assertEqual(set(('original','symmetric','asymmetric')) -
                         set(summary), set())
        self.assertTrue((out.lower_asymmetric <= out.upper_asymmetric).all())

    def test_no_future_actuals_affect_first_day(self):
        df = self.sample()
        base, _ = compare(df, warmup_days=4, window_days=4)
        altered = df.copy()
        altered.loc[altered.index >= 4 * 48, 'actual'] = 10000.
        changed, _ = compare(altered, warmup_days=4, window_days=4)
        self.assertEqual(base.iloc[0].lower_radius_asymmetric,
                         changed.iloc[0].lower_radius_asymmetric)
        self.assertEqual(base.iloc[0].upper_radius_asymmetric,
                         changed.iloc[0].upper_radius_asymmetric)

    def test_published_history_cutoff(self):
        out, _ = compare(self.sample(), warmup_days=4, window_days=4)
        last = pd.to_datetime(out.asymmetric_latest_history_utc, utc=True)
        issue = pd.to_datetime(out.forecast_issue_utc, utc=True)
        self.assertTrue(((last + pd.Timedelta(hours=48)) <= issue).all())


if __name__ == '__main__':
    unittest.main()
